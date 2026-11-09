"""Fenêtre simple avec capture et traitements indépendants."""

from __future__ import annotations

from collections import deque
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass
import time

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QImage, QKeySequence, QPainter, QShortcut
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout,
                               QLabel, QFileDialog, QLineEdit, QMainWindow, QMessageBox,
                               QPushButton, QSizePolicy, QSplitter, QStackedWidget,
                               QToolButton, QVBoxLayout, QWidget)

from capture import CaptureError, FirefoxCapture, list_windows
from game import GameView
from sequence import SequenceReader
from super_resolution import (ModelError, available_model_names, create_processor,
                               supported_scales)


RESOLUTIONS = {"320 × 180": (320, 180), "480 × 270": (480, 270), "640 × 360": (640, 360)}
CAPTURE_RATES = {"5 FPS": 5, "10 FPS": 10, "15 FPS": 15, "30 FPS": 30, "60 FPS": 60}


def image_metrics(reference: np.ndarray, result: np.ndarray) -> tuple[float, float]:
    """Calcule PSNR et SSIM sur deux images BGR de mêmes dimensions."""
    if reference.shape != result.shape:
        raise ValueError(f"Images incompatibles: {reference.shape} et {result.shape}.")
    reference_f = reference.astype(np.float64)
    result_f = result.astype(np.float64)
    mse = np.mean((reference_f - result_f) ** 2)
    psnr = float("inf") if mse == 0 else float(10 * np.log10((255 ** 2) / mse))
    values = []
    for channel in range(3):
        x, y = reference_f[:, :, channel], result_f[:, :, channel]
        mean_x, mean_y = x.mean(), y.mean()
        variance_x, variance_y = x.var(), y.var()
        covariance = np.mean((x - mean_x) * (y - mean_y))
        c1, c2 = 6.5025, 58.5225
        values.append(((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) /
                      ((mean_x ** 2 + mean_y ** 2 + c1) *
                       (variance_x + variance_y + c2)))
    return psnr, float(np.mean(values))


def pixel_difference_metrics(first: np.ndarray, second: np.ndarray) -> tuple[float, int, float]:
    """Mesure l'écart brut entre deux reconstructions de mêmes dimensions."""
    if first.shape != second.shape:
        raise ValueError(f"Images incompatibles: {first.shape} et {second.shape}.")
    difference = np.abs(first.astype(np.int16) - second.astype(np.int16))
    return (float(difference.mean()), int(difference.max()),
            float(np.count_nonzero(difference) / difference.size * 100))


def difference_image(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Crée une visualisation volontairement amplifiée des différences."""
    if first.shape != second.shape:
        raise ValueError("Les deux images doivent avoir les mêmes dimensions.")
    difference = cv2.absdiff(first, second)
    return cv2.convertScaleAbs(difference, alpha=4.0)


def prepare_fixed_image(image: np.ndarray, scale: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Réduit une photo selon le facteur choisi et conserve sa référence."""
    height, width = image.shape[:2]
    reduced = cv2.resize(image, (max(1, width // scale), max(1, height // scale)),
                         interpolation=cv2.INTER_AREA)
    return reduced, image


@dataclass(frozen=True)
class Frame:
    """Image capturée une seule fois, avec son identifiant et son instant."""

    image: np.ndarray
    sequence: int
    captured_at: float


def resize_for_model(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Redimensionne sans déformation par ajustement puis recadrage central."""
    target_width, target_height = size
    height, width = image.shape[:2]
    scale = max(target_width / width, target_height / height)
    resized = cv2.resize(image, (round(width * scale), round(height * scale)),
                         interpolation=cv2.INTER_AREA)
    top = (resized.shape[0] - target_height) // 2
    left = (resized.shape[1] - target_width) // 2
    return np.ascontiguousarray(resized[top:top + target_height, left:left + target_width])


class RegionPicker(QDialog):
    def __init__(self, screen_geometry: QRect, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Région du jeu")
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
        self.setWindowOpacity(0.35)
        self.setGeometry(screen_geometry)
        self.start: QPoint | None = None
        self.end: QPoint | None = None
        self.setCursor(Qt.CursorShape.CrossCursor)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.start = self.end = event.position().toPoint()
            self.update()

    def mouseMoveEvent(self, event) -> None:
        if self.start is not None:
            self.end = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.start and self.end:
            self.accept()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.reject()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        if self.start and self.end:
            rectangle = QRect(self.start, self.end).normalized()
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.fillRect(rectangle, Qt.GlobalColor.transparent)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            painter.setPen(Qt.GlobalColor.white)
            painter.drawRect(rectangle)

    def region(self) -> tuple[int, int, int, int] | None:
        if not self.start or not self.end:
            return None
        rectangle = QRect(self.start, self.end).normalized()
        if rectangle.width() < 2 or rectangle.height() < 2:
            return None
        return (self.x() + rectangle.x(), self.y() + rectangle.y(),
                rectangle.width(), rectangle.height())


class ImagePanel(QWidget):
    zoom_changed = Signal(float)

    def __init__(self, title: str, device: str = "") -> None:
        super().__init__()
        self.title = QLabel(title)
        self.stats = QLabel("En attente d'une image...")
        self.image = QImage()
        self.pixels: np.ndarray | None = None
        self.finished: deque[float] = deque()
        self.fps_value = 0
        self.device = device
        self.zoom = 1.0
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.stats.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.stats.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self.title)
        layout.addWidget(self.stats)

    def set_result(self, image: np.ndarray, milliseconds: float, now: float,
                   input_size: tuple[int, int]) -> None:
        self.pixels = np.ascontiguousarray(image)
        height, width = self.pixels.shape[:2]
        self.image = QImage(self.pixels.data, width, height, self.pixels.strides[0],
                            QImage.Format.Format_BGR888)
        self.finished.append(now)
        while self.finished and now - self.finished[0] > 1:
            self.finished.popleft()
        fps = len(self.finished)
        self.fps_value = fps
        output = f"{width} × {height}"
        suffix = f" — {self.device}" if self.device else ""
        self.stats.setText(f"{self.title.text()} — {fps} FPS affichés — "
                           f"{milliseconds:.1f} ms — {input_size[0]}×{input_size[1]} → {output}{suffix}")
        self.update()

    def set_title(self, title: str) -> None:
        self.title.setText(title)

    def set_static(self, image: np.ndarray, title: str) -> None:
        self.pixels = np.ascontiguousarray(image)
        height, width = self.pixels.shape[:2]
        self.image = QImage(self.pixels.data, width, height, self.pixels.strides[0],
                            QImage.Format.Format_BGR888)
        self.set_title(title)
        self.stats.setText(f"{width} × {height}")
        self.update()

    def set_zoom(self, zoom: float) -> None:
        self.zoom = max(1.0, min(8.0, zoom))
        self.update()

    def wheelEvent(self, event) -> None:
        self.zoom_changed.emit(1.25 if event.angleDelta().y() > 0 else 0.8)
        event.accept()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        if not self.image.isNull():
            scale = min(self.width() / self.image.width(), self.height() / self.image.height())
            width, height = self.image.width() * scale * self.zoom, self.image.height() * scale * self.zoom
            scaled = self.image.scaled(int(width), int(height), Qt.AspectRatioMode.KeepAspectRatio,
                                       Qt.TransformationMode.FastTransformation)
            painter.drawImage((self.width() - width) / 2, (self.height() - height) / 2, scaled)


class ModelSelectionDialog(QDialog):
    """Sélectionne uniquement deux modèles et un facteur compatibles."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Sup-Eirb — configuration")
        names = available_model_names()
        self.first = QComboBox()
        self.second = QComboBox()
        self.first.addItems(names)
        self.second.addItems(names)
        if len(names) > 1:
            self.second.setCurrentIndex(1)
        self.scale = QComboBox()
        self.first.currentTextChanged.connect(self._update_scales)
        self.second.currentTextChanged.connect(self._update_scales)
        self._update_scales()
        form = QVBoxLayout(self)
        if len(names) < 2:
            form.addWidget(QLabel(
                "Moins de deux modèles fonctionnels ont été trouvés dans corrections.\n"
                "Ajoutez les poids entraînés et leurs adaptateurs validés, puis relancez."
            ))
        form.addWidget(QLabel("Premier modèle"))
        form.addWidget(self.first)
        form.addWidget(QLabel("Deuxième modèle"))
        form.addWidget(self.second)
        form.addWidget(QLabel("Facteur d'agrandissement"))
        form.addWidget(self.scale)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                   QDialogButtonBox.StandardButton.Cancel)
        self.ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.ok_button.setEnabled(len(names) >= 2 and bool(self.scale.count()))
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        form.addWidget(buttons)

    def _update_scales(self) -> None:
        if not self.first.currentText() or not self.second.currentText():
            self.scale.clear()
            if hasattr(self, "ok_button"):
                self.ok_button.setEnabled(False)
            return
        common = sorted(set(supported_scales(self.first.currentText())) &
                        set(supported_scales(self.second.currentText())))
        current = self.scale.currentData()
        self.scale.clear()
        for value in common:
            self.scale.addItem(f"×{value}", value)
        if current in common:
            self.scale.setCurrentIndex(common.index(current))
        if hasattr(self, "ok_button"):
            self.ok_button.setEnabled(self.first.currentText() != self.second.currentText()
                                      and bool(common))

    def _accept(self) -> None:
        if self.first.currentText() == self.second.currentText() or not self.scale.count():
            QMessageBox.warning(self, "Modèles indisponibles",
                                "Choisissez deux modèles différents avec un facteur commun.")
            return
        self.accept()

    def selection(self) -> tuple[str, str, int]:
        return self.first.currentText(), self.second.currentText(), int(self.scale.currentData())


class MainWindow(QMainWindow):
    bicubic_ready = Signal(object)
    espcn_ready = Signal(object)
    fixed_ready = Signal(object)

    def __init__(self, model_names: tuple[str, str], scale: int) -> None:
        super().__init__()
        self.closing = False
        self.setMinimumSize(0, 0)
        self.model_names = model_names
        self.scale = scale
        self.setWindowTitle(f"Sup-Eirb — {model_names[0]} / {model_names[1]}")
        self.resize(1100, 800)
        self.game = GameView()
        self.external: FirefoxCapture | None = None
        self.source = self.game
        self.status = QLabel("Choisissez une adresse de jeu ou capturez Firefox.")
        self.welcome = QLabel("Ouvrez un jeu, une vidéo ou une image pour commencer.")
        self.url = QLineEdit()
        self.url.setPlaceholderText("Adresse du jeu : https://...")
        open_button = QPushButton("Ouvrir")
        firefox_button = QPushButton("Capturer un jeu Firefox")
        self.pause_button = QPushButton("Temps réel")
        self.image_button = QPushButton("Image")
        self.sequence_button = QPushButton("Séquence")
        self.models_button = QPushButton("Modèles")
        self.play_button = QPushButton("Lecture")
        self.previous_button = QPushButton("<")
        self.previous_button.setToolTip("Image précédente")
        self.next_button = QPushButton(">")
        self.next_button.setToolTip("Image suivante")
        self.restart_button = QPushButton("Début")
        self.playback_rate = QComboBox()
        self.playback_rate.addItems(["5 FPS", "10 FPS", "15 FPS", "30 FPS", "60 FPS"])
        self.playback_rate.setCurrentText("10 FPS")
        self.expand_button = QAction("Agrandir", self)
        self.diff_button = QAction("Différences", self)
        self.zoom_reset_button = QAction("Zoom 1×", self)
        open_button.clicked.connect(self.open_url)
        self.url.returnPressed.connect(self.open_url)
        firefox_button.clicked.connect(self.open_firefox)
        self.pause_button.clicked.connect(self.toggle_sync)
        self.image_button.clicked.connect(self.open_image)
        self.sequence_button.clicked.connect(self.open_sequence)
        self.models_button.clicked.connect(self.change_models)
        self.play_button.clicked.connect(self.toggle_sequence_playback)
        self.previous_button.clicked.connect(self.previous_sequence_frame)
        self.next_button.clicked.connect(self.next_sequence_frame)
        self.restart_button.clicked.connect(self.restart_sequence)
        self.playback_rate.currentTextChanged.connect(self.change_playback_rate)
        self.expand_button.triggered.connect(self.toggle_expanded)
        self.diff_button.triggered.connect(self.toggle_difference)
        self.zoom_reset_button.triggered.connect(self.reset_zoom)
        self.resolution = QComboBox()
        self.resolution.addItems(RESOLUTIONS)
        self.resolution.setCurrentText("320 × 180")
        self.capture_rate = QComboBox()
        self.capture_rate.addItems(CAPTURE_RATES)
        self.capture_rate.setCurrentText("60 FPS")
        self.capture_rate.currentTextChanged.connect(self.change_capture_rate)
        more_button = QToolButton()
        more_button.setText("Plus")
        more_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        more_button.setMenu(self._create_more_menu())
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Jeu :"))
        source_row.addWidget(self.url, 1)
        for widget in (open_button, firefox_button, self.pause_button, self.image_button,
                       self.sequence_button):
            source_row.addWidget(widget)

        comparison_row = QHBoxLayout()
        for widget in (self.models_button, self.play_button, self.previous_button,
                       self.next_button, self.restart_button):
            comparison_row.addWidget(widget)
        comparison_row.addWidget(self.playback_rate)
        self.playback_rate.setToolTip("Vitesse de lecture")
        comparison_row.addWidget(self.resolution)
        self.resolution.setToolTip("Résolution de traitement")
        comparison_row.addWidget(self.capture_rate)
        self.capture_rate.setToolTip("Fréquence de capture")
        comparison_row.addWidget(more_button)
        controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addLayout(source_row)
        controls_layout.addLayout(comparison_row)

        self.left = ImagePanel(f"{model_names[0]} ×{scale}")
        self.right = ImagePanel(f"{model_names[1]} ×{scale}")
        self.original_panel = ImagePanel("Image originale")
        self.source_stack = QStackedWidget()
        self.source_stack.addWidget(self.game)
        self.source_stack.addWidget(self.original_panel)
        bottom = QWidget()
        results = QHBoxLayout(bottom)
        results.setContentsMargins(0, 0, 0, 0)
        results.addWidget(self.left)
        results.addWidget(self.right)
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.source_stack)
        splitter.addWidget(bottom)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.addLayout(controls_layout)
        layout.addWidget(splitter, 1)
        self.metrics = QLabel("Capture : -- FPS | Modèle 1 : -- FPS | Modèle 2 : -- FPS")
        layout.addWidget(self.metrics)
        layout.addWidget(self.welcome)
        layout.addWidget(self.status)
        self.setCentralWidget(root)
        self.game.status_changed.connect(self.status.setText)
        self.game.fullscreen_requested.connect(self.set_fullscreen)

        self.bicubic_ready.connect(self._show_bicubic)
        self.espcn_ready.connect(self._show_espcn)
        self.fixed_ready.connect(self._show_fixed)
        self.bicubic_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-1")
        self.espcn_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-2")
        self.fixed_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fixed-image")
        self.bicubic_future: Future | None = None
        self.espcn_future: Future | None = None
        self.fixed_future: Future | None = None
        self.processors = None
        self.latest_frame: Frame | None = None
        self.sync_frame: Frame | None = None
        self.sequence = 0
        self.capture_times: deque[float] = deque()
        self.sync_mode = False
        self.fixed_mode = False
        self.expanded = False
        self.showing_difference = False
        self.fixed_original: np.ndarray | None = None
        self.fixed_bicubic: np.ndarray | None = None
        self.fixed_espcn: np.ndarray | None = None
        self.fixed_size: tuple[int, int] | None = None
        self.sequence_reader: SequenceReader | None = None
        self.sequence_index = 0
        self.sequence_mode = False
        self.sequence_playing = False
        self.zoom = 1.0
        self.left.zoom_changed.connect(self.change_zoom)
        self.right.zoom_changed.connect(self.change_zoom)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._capture_tick)
        try:
            self.processors = (create_processor(model_names[0], scale),
                               create_processor(model_names[1], scale))
            devices = " / ".join(processor.device for processor in self.processors)
            self.status.setText(f"Modèles prêts ({devices}). Ouvrez une source pour commencer.")
            self.timer.start(self._capture_interval())
        except ModelError as exc:
            self.status.setText(f"Erreur CUDA : {exc}")
        QShortcut(QKeySequence("F11"), self, activated=self.toggle_fullscreen)
        QTimer.singleShot(0, lambda: splitter.setSizes([max(1, self.height() // 3),
                                                         max(1, self.height() * 2 // 3)]))

    def _create_more_menu(self):
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        menu.addAction(self.expand_button)
        menu.addAction(self.diff_button)
        menu.addAction(self.zoom_reset_button)
        return menu

    def _capture_interval(self) -> int:
        return round(1000 / CAPTURE_RATES[self.capture_rate.currentText()])

    def change_capture_rate(self) -> None:
        if self.timer.isActive():
            self.timer.start(self._capture_interval())

    def change_playback_rate(self) -> None:
        if self.sequence_mode and self.sequence_playing:
            self.timer.start(round(1000 / int(self.playback_rate.currentText().split()[0])))

    def change_models(self) -> None:
        dialog = ModelSelectionDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        first, second, scale = dialog.selection()
        self._replacement = MainWindow((first, second), scale)
        self._replacement.show()
        self.close()

    def open_sequence(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Ouvrir une séquence d'images")
        if not folder:
            return
        try:
            reader = SequenceReader(folder)
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self._stop_external()
        self.sequence_reader = reader
        self.sequence_index = 0
        self.sequence_mode = True
        self.sequence_playing = False
        self.fixed_mode = False
        self.sync_mode = True
        self.timer.stop()
        self.source_stack.setCurrentWidget(self.original_panel)
        self.welcome.hide()
        self._show_sequence_frame()

    def _show_sequence_frame(self) -> None:
        if self.sequence_reader is None:
            return
        image = self.sequence_reader.read(self.sequence_index)
        self.original_panel.set_static(
            image, f"Frame {self.sequence_index + 1}/{len(self.sequence_reader.paths)}")
        self.sequence += 1
        self.latest_frame = Frame(image, self.sequence, time.perf_counter())
        self._schedule_workers(self.latest_frame)
        self.status.setText(f"Frame {self.sequence_index + 1}/{len(self.sequence_reader.paths)}")

    def toggle_sequence_playback(self) -> None:
        if not self.sequence_mode:
            self.status.setText("Ouvrez d'abord une séquence d'images.")
            return
        self.sequence_playing = not self.sequence_playing
        if self.sequence_playing:
            self.play_button.setText("Pause")
            self.timer.start(round(1000 / int(self.playback_rate.currentText().split()[0])))
        else:
            self.play_button.setText("Lecture")
            self.timer.stop()

    def next_sequence_frame(self) -> None:
        if self.sequence_reader is None:
            return
        self.sequence_index = min(self.sequence_index + 1, len(self.sequence_reader.paths) - 1)
        self._show_sequence_frame()

    def previous_sequence_frame(self) -> None:
        if self.sequence_reader is None:
            return
        self.sequence_index = max(0, self.sequence_index - 1)
        self._show_sequence_frame()

    def restart_sequence(self) -> None:
        if self.sequence_reader is None:
            return
        self.sequence_index = 0
        self._show_sequence_frame()

    def open_url(self) -> None:
        address = self.url.text().strip()
        if not address:
            self.status.setText("Saisissez une adresse http:// ou https://.")
            return
        self._stop_external()
        self.source = self.game
        self.fixed_mode = False
        self.sequence_mode = False
        self.source_stack.setCurrentWidget(self.game)
        self.welcome.hide()
        self.game.show()
        self.game.open(address)

    def open_firefox(self) -> None:
        self._stop_external()
        try:
            windows = list_windows({int(self.winId())})
            if not windows:
                raise CaptureError("Aucune fenêtre X11 visible.")
            labels = [f"{window.title} ({window.width}×{window.height})" for window in windows]
            from PySide6.QtWidgets import QInputDialog
            value, ok = QInputDialog.getItem(self, "Fenêtres X11", "Fenêtre Firefox :", labels, 0, False)
            if not ok:
                return
            window = windows[labels.index(value)]
            answer = QMessageBox.question(
                self, "Zone de capture", "Capturer toute la fenêtre Firefox ?\n"
                "Choisissez Non pour dessiner une région.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No |
                QMessageBox.StandardButton.Cancel)
            if answer == QMessageBox.StandardButton.Cancel:
                return
            region = None
            if answer == QMessageBox.StandardButton.No:
                self.hide()
                picker = RegionPicker(QApplication.primaryScreen().availableGeometry())
                accepted = picker.exec()
                region = picker.region() if accepted else None
                self.show()
                if region is None:
                    return
            self.external = FirefoxCapture(window, region)
            self.source = self.external
            self.fixed_mode = False
            self.sequence_mode = False
            self.source_stack.setCurrentWidget(self.game)
            self.welcome.hide()
            self.status.setText("Capture Firefox active. Le jeu reste contrôlé dans Firefox.")
        except (CaptureError, OSError) as exc:
            self.status.setText(f"Capture Firefox indisponible : {exc}")

    def open_image(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "Ouvrir une image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not filename:
            return
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        if image is None:
            self.status.setText("Image impossible à ouvrir.")
            return
        reduced, original = prepare_fixed_image(image, self.scale)
        self._stop_external()
        self.fixed_mode = True
        self.sync_mode = True
        self.timer.stop()
        self.source_stack.setCurrentWidget(self.original_panel)
        self.original_panel.set_static(original, "Image originale — référence")
        self.fixed_original = original
        self.fixed_size = (reduced.shape[1], reduced.shape[0])
        self.welcome.hide()
        self.status.setText(f"Image fixe chargée. Calcul de {self.model_names[0]} et "
                            f"{self.model_names[1]}...")
        self.fixed_future = self.fixed_executor.submit(self._process_fixed, reduced, original)
        self.fixed_future.add_done_callback(self.fixed_ready.emit)

    def _process_fixed(self, reduced: np.ndarray, original: np.ndarray):
        first, second = self.processors
        first_image, first_ms = first.upscale(reduced)
        second_image, second_ms = second.upscale(reduced)
        target = (original.shape[1], original.shape[0])
        first_image = cv2.resize(first_image, target, interpolation=cv2.INTER_AREA)
        second_image = cv2.resize(second_image, target, interpolation=cv2.INTER_AREA)
        return (first_image, second_image, first_ms, second_ms,
                image_metrics(original, first_image), image_metrics(original, second_image))

    def _show_fixed(self, future: Future) -> None:
        self.fixed_future = None
        if self.closing:
            return
        try:
            bicubic, espcn, bicubic_ms, espcn_ms, bicubic_metrics, espcn_metrics = future.result()
        except (CancelledError, ModelError) as exc:
            if not isinstance(exc, CancelledError):
                self.status.setText(f"Erreur image fixe : {exc}")
            return
        self.fixed_bicubic, self.fixed_espcn = bicubic, espcn
        self.left.set_static(bicubic, f"{self.model_names[0]} ×{self.scale}")
        self.right.set_static(espcn, f"{self.model_names[1]} ×{self.scale}")
        mae, maximum, different = pixel_difference_metrics(bicubic, espcn)
        self.left.stats.setText(f"PSNR : {bicubic_metrics[0]:.2f} dB | SSIM : {bicubic_metrics[1]:.4f} | "
                            f"{bicubic_ms:.1f} ms")
        self.right.stats.setText(f"PSNR : {espcn_metrics[0]:.2f} dB | SSIM : {espcn_metrics[1]:.4f} | "
                            f"{espcn_ms:.1f} ms")
        self.status.setText("Image fixe prête. Utilisez la molette sur un panneau pour zoomer.")
        self.metrics.setText(f"Référence : {self.fixed_original.shape[1]}×{self.fixed_original.shape[0]} | "
                             f"Réduction : {self.fixed_size[0]}×{self.fixed_size[1]} | "
                             f"Différence {self.model_names[1]}/{self.model_names[0]} : "
                             f"MAE {mae:.2f} | max {maximum} | "
                             f"{different:.2f}% pixels différents")

    def change_zoom(self, multiplier: float) -> None:
        self.zoom = max(1.0, min(8.0, self.zoom * multiplier))
        self.left.set_zoom(self.zoom)
        self.right.set_zoom(self.zoom)

    def reset_zoom(self) -> None:
        self.zoom = 1.0
        self.left.set_zoom(self.zoom)
        self.right.set_zoom(self.zoom)

    def toggle_expanded(self) -> None:
        self.expanded = not self.expanded
        splitter = self.centralWidget().findChild(QSplitter)
        if splitter is not None:
            splitter.setSizes([0, splitter.height()] if self.expanded else
                              [max(1, splitter.height() // 3), max(1, splitter.height() * 2 // 3)])
        self.expand_button.setText("Affichage normal" if self.expanded else "Comparaison agrandie")

    def toggle_difference(self) -> None:
        if self.fixed_bicubic is None or self.fixed_espcn is None:
            self.status.setText("Chargez d'abord une image fixe.")
            return
        self.showing_difference = not self.showing_difference
        if self.showing_difference:
            self.right.set_static(difference_image(self.fixed_bicubic, self.fixed_espcn),
                                  f"Différences amplifiées | {self.model_names[1]} - "
                                  f"{self.model_names[0]}")
            self.diff_button.setText(f"Afficher {self.model_names[1]}")
        else:
            self.right.set_static(self.fixed_espcn, f"{self.model_names[1]} ×{self.scale}")
            self.diff_button.setText("Afficher les différences")

    def _stop_external(self) -> None:
        if self.external:
            self.external.close()
            self.external = None

    def _capture_tick(self) -> None:
        if self.sequence_mode:
            if self.sequence_reader is not None and self.sequence_index + 1 < len(self.sequence_reader.paths):
                self.sequence_index += 1
                self._show_sequence_frame()
            else:
                self.sequence_playing = False
                self.play_button.setText("Lecture")
                self.timer.stop()
            return
        if self.processors is None or self.sync_mode:
            return
        self._capture_and_schedule()

    def _capture_and_schedule(self) -> None:
        try:
            image = self.source.capture()
        except (CaptureError, OSError) as exc:
            self.status.setText(f"Capture interrompue : {exc}")
            return
        if image is None:
            return
        now = time.perf_counter()
        self.sequence += 1
        frame = Frame(image, self.sequence, now)
        self.latest_frame = frame
        self.capture_times.append(now)
        while self.capture_times and now - self.capture_times[0] > 1:
            self.capture_times.popleft()
        self._update_metrics()
        self._schedule_workers(frame)

    def _schedule_workers(self, frame: Frame) -> None:
        size = RESOLUTIONS[self.resolution.currentText()]
        low = resize_for_model(frame.image, size)
        if self.bicubic_future is None:
            self.bicubic_future = self.bicubic_executor.submit(self._process_bicubic, frame, low, size)
            self.bicubic_future.add_done_callback(self.bicubic_ready.emit)
        if self.espcn_future is None:
            self.espcn_future = self.espcn_executor.submit(self._process_espcn, frame, low, size)
            self.espcn_future.add_done_callback(self.espcn_ready.emit)

    def _process_bicubic(self, frame: Frame, low: np.ndarray, size: tuple[int, int]):
        result, milliseconds = self.processors[0].upscale(low)
        return frame, result, milliseconds, size

    def _process_espcn(self, frame: Frame, low: np.ndarray, size: tuple[int, int]):
        result, milliseconds = self.processors[1].upscale(low)
        return frame, result, milliseconds, size

    def _show_bicubic(self, future: Future) -> None:
        self.bicubic_future = None
        if self.closing:
            return
        try:
            frame, image, milliseconds, size = future.result()
        except CancelledError:
            return
        except ModelError as exc:
            self.status.setText(f"Erreur {self.model_names[0]} : {exc}")
            return
        now = time.perf_counter()
        self.left.set_result(image, milliseconds, now, size)
        self._update_metrics()
        self._schedule_next(self.bicubic_executor, "bicubic", frame)

    def _show_espcn(self, future: Future) -> None:
        self.espcn_future = None
        if self.closing:
            return
        try:
            frame, image, milliseconds, size = future.result()
        except CancelledError:
            return
        except ModelError as exc:
            self.status.setText(f"Erreur {self.model_names[1]} : {exc}")
            return
        now = time.perf_counter()
        self.right.set_result(image, milliseconds, now, size)
        self._update_metrics()
        self._schedule_next(self.espcn_executor, "espcn", frame)

    def _update_metrics(self) -> None:
        capture_fps = len(self.capture_times)
        self.metrics.setText(f"Capture réelle : {capture_fps} FPS | "
                            f"{self.model_names[0]} affiché : {self.left.fps_value} FPS | "
                            f"{self.model_names[1]} affiché : {self.right.fps_value} FPS | "
                            f"Mode : {'synchronisé' if self.sync_mode else 'temps réel'}")

    def _schedule_next(self, executor: ThreadPoolExecutor, name: str, processed: Frame) -> None:
        frame = self.sync_frame if self.sync_mode else self.latest_frame
        if frame is None or frame.sequence == processed.sequence:
            return
        size = RESOLUTIONS[self.resolution.currentText()]
        low = resize_for_model(frame.image, size)
        if name == "bicubic":
            self.bicubic_future = executor.submit(self._process_bicubic, frame, low, size)
            self.bicubic_future.add_done_callback(self.bicubic_ready.emit)
        else:
            self.espcn_future = executor.submit(self._process_espcn, frame, low, size)
            self.espcn_future.add_done_callback(self.espcn_ready.emit)

    def toggle_sync(self) -> None:
        if self.sync_mode:
            self.sync_mode = False
            self.sync_frame = None
            self.pause_button.setText("Pause / Comparer la même image")
            self.status.setText("Mode temps réel actif.")
            self.timer.start(self._capture_interval())
            return
        self._capture_and_schedule()
        if self.latest_frame is None:
            return
        self.sync_mode = True
        self.sync_frame = self.latest_frame
        self.timer.stop()
        self.pause_button.setText("Reprendre le temps réel")
        self.status.setText(f"Comparaison synchronisée sur l'image {self.sync_frame.sequence}.")

    def set_fullscreen(self, enabled: bool) -> None:
        self.showFullScreen() if enabled else self.showNormal()

    def toggle_fullscreen(self) -> None:
        self.set_fullscreen(not self.isFullScreen())

    def closeEvent(self, event) -> None:
        self.closing = True
        self.status.setText("Fermeture de Sup-Eirb...")
        self.timer.stop()
        for future in (self.bicubic_future, self.espcn_future, self.fixed_future):
            if future:
                future.cancel()
        self.bicubic_executor.shutdown(wait=True, cancel_futures=True)
        self.espcn_executor.shutdown(wait=True, cancel_futures=True)
        self.fixed_executor.shutdown(wait=True, cancel_futures=True)
        self._stop_external()
        self.game.close()
        event.accept()


def run() -> int:
    app = QApplication.instance() or QApplication([])
    dialog = ModelSelectionDialog()
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return 0
    first, second, scale = dialog.selection()
    window = MainWindow((first, second), scale)
    window.show()
    return app.exec()
