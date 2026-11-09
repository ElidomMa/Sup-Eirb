"""Fenêtre simple avec capture et traitements indépendants."""

from __future__ import annotations

from collections import deque
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
import time

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, QRectF, Qt, QTimer, Signal
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
from pipeline import Frame, Request, LatestRequest, process_pair, resize_for_model


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
    height, width = height - height % scale, width - width % scale
    if height == 0 or width == 0:
        raise ValueError("Image trop petite pour ce facteur.")
    image = image[:height, :width]
    reduced = cv2.resize(image, (width // scale, height // scale),
                         interpolation=cv2.INTER_AREA)
    return reduced, image


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
        self.paint_ms = 0.0
        self._pending_display = False
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
        layout.addStretch(1)

    def set_result(self, image: np.ndarray, milliseconds: float, now: float,
                   input_size: tuple[int, int], network_ms: float | None = None,
                   sequence: int | None = None) -> None:
        self.pixels = np.ascontiguousarray(image)
        height, width = self.pixels.shape[:2]
        self.image = QImage(self.pixels.data, width, height, self.pixels.strides[0],
                            QImage.Format.Format_BGR888)
        fps = self.display_fps(now)
        self._pending_display = True
        output = f"{width} × {height}"
        suffix = f" — {self.device}" if self.device else ""
        network = milliseconds if network_ms is None else network_ms
        rate = 1000 / network if network > 0 else 0
        frame = f" — frame {sequence}" if sequence is not None else ""
        self.stats.setText(f"{fps:.2f} FPS affichés — réseau {network:.1f} ms ({rate:.1f} FPS théoriques) — "
                           f"traitement {milliseconds:.1f} ms — {input_size[0]}×{input_size[1]} → {output}{suffix}{frame}")
        self.update()

    def display_fps(self, now):
        while self.finished and now - self.finished[0] > 10:
            self.finished.popleft()
        self.fps_value = ((len(self.finished) - 1) / max(now - self.finished[0], 1e-6)
                          if len(self.finished) > 1 else 0.0)
        return self.fps_value

    def set_title(self, title: str) -> None:
        self.title.setText(title)

    def set_static(self, image: np.ndarray, title: str) -> None:
        self._pending_display = False
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
        started = time.perf_counter()
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        if not self.image.isNull():
            area = QRectF(self.rect()).adjusted(0, self.stats.geometry().bottom() + 6, 0, 0)
            if area.height() > 0:
                scale = min(area.width() / self.image.width(), area.height() / self.image.height()) * self.zoom
                width, height = self.image.width() * scale, self.image.height() * scale
                target = QRectF(area.center().x() - width / 2, area.center().y() - height / 2, width, height)
                painter.setClipRect(area)
                painter.drawImage(target, self.image)
                if self._pending_display:
                    self.finished.append(time.perf_counter())
                    self._pending_display = False
        self.paint_ms = (time.perf_counter() - started) * 1000


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
    work_ready = Signal(object)
    source_ready = Signal(object)

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
        self.source_active = False
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

        self.work_ready.connect(self._work_done)
        self.source_ready.connect(self._source_done)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
        self.capture_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="capture")
        self.future: Future | None = None
        self.capture_future: Future | None = None
        self.requests = LatestRequest()
        self.generation = 0
        self.processors = None
        self.model_change = (model_names, scale)
        self.models_button.setEnabled(False)
        self.last_displayed_sequence = None
        self.displayed_frames = (None, None)
        self._sequence_loaded_index = None
        self.last_comparison = None
        self.latest_frame: Frame | None = None
        self.sync_frame: Frame | None = None
        self.sequence = 0
        self.capture_times: deque[float] = deque()
        self.sync_mode = False
        self.fixed_mode = False
        self.expanded = False
        self.showing_difference = False
        self.fixed_original: np.ndarray | None = None
        self.fixed_first: np.ndarray | None = None
        self.fixed_second: np.ndarray | None = None
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
        self.resolution.currentTextChanged.connect(self.change_resolution)
        for index, description in enumerate(("Performance", "Équilibré", "Qualité")):
            self.resolution.setItemData(index, description, Qt.ItemDataRole.ToolTipRole)
        self.fps_timer = QTimer(self)
        self.fps_timer.timeout.connect(self._update_metrics)
        self.fps_timer.start(250)
        self._dispatch()
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
        self._invalidate()
        self.model_names, self.scale = (first, second), scale
        self.model_change = (self.model_names, scale)
        self.models_button.setEnabled(False)
        self.left.set_title(f"{first} ×{scale}")
        self.right.set_title(f"{second} ×{scale}")
        self.setWindowTitle(f"Sup-Eirb — {first} / {second}")
        self._dispatch()

    def _invalidate(self):
        self.generation += 1
        self.requests.clear()
        self.last_comparison = None
        self.last_displayed_sequence = None
        self.fixed_first = self.fixed_second = None
        self.showing_difference = False
        self.diff_button.setText("Différences")
        for panel in (self.left, self.right):
            panel.image = QImage()
            panel.pixels = None
            panel.finished.clear()
            panel.fps_value = 0.0
            panel.update()
        self.displayed_frames = (None, None)

    def change_resolution(self):
        self._invalidate()
        if self.latest_frame is not None:
            self._schedule_workers(self.latest_frame)

    def _load_models(self, selection):
        self._release_models()
        names, scale = selection
        loaded = []
        try:
            for name in names:
                loaded.append(create_processor(name, scale))
        except Exception:
            for processor in loaded:
                processor.close()
            raise
        self.processors = tuple(loaded)
        return "loaded", selection

    def _release_models(self):
        previous, self.processors = self.processors, None
        for processor in previous or ():
            processor.close()

    def _dispatch(self):
        if self.closing or self.future is not None:
            return
        if self.model_change is not None:
            selection, self.model_change = self.model_change, None
            self.status.setText("Chargement des modèles…")
            self.future = self.executor.submit(self._load_models, selection)
        elif self.processors is not None:
            request = self.requests.take()
            if request is None:
                return
            self.future = self.executor.submit(self._process_request, request)
        else:
            return
        self.future.add_done_callback(self.work_ready.emit)

    def _process_request(self, request):
        result = process_pair(self.processors, request)
        if request.fixed:
            result.metrics = tuple(image_metrics(result.reference, image) for image in result.images)
            result.difference = pixel_difference_metrics(*result.images)
        return "comparison", result

    def _work_done(self, future):
        if future is not self.future:
            return
        self.future = None
        if self.closing:
            return
        try:
            kind, result = future.result()
        except CancelledError:
            return
        except Exception as exc:
            self.status.setText(f"Traitement interrompu : {exc}")
            self.timer.stop()
            self.requests.clear()
            self.models_button.setEnabled(True)
            if self.model_change is not None:
                self._dispatch()
            return
        if kind == "loaded":
            self.models_button.setEnabled(True)
            for panel, processor in zip((self.left, self.right), self.processors):
                panel.device = f"{processor.device} / {'FP16' if processor.use_half else 'FP32'}"
            devices = " / ".join(panel.device for panel in (self.left, self.right))
            notice = " CUDA indisponible : exécution CPU lente." if any(
                processor.device == "cpu" for processor in self.processors) else ""
            self.status.setText(f"Modèles prêts : {devices}.{notice}")
            if not self.sync_mode:
                self.timer.start(self._capture_interval())
            if self.latest_frame is not None:
                self._schedule_workers(self.latest_frame)
        else:
            request = result.request
            current = request.generation == self.generation
            if self.sync_mode and not self.sequence_playing and self.latest_frame is not None:
                current = current and request.frame.sequence == self.latest_frame.sequence
            if current:
                self._show_comparison(result)
        self._dispatch()

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
        self._invalidate()
        self.latest_frame = None
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
        if self.capture_future is not None:
            return
        index = self.sequence_index
        if self._sequence_loaded_index == (self.generation, index):
            return
        self._submit_source(self.sequence_reader.read, (index,), "sequence", index)

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
        self._invalidate()
        self.latest_frame = None
        self.sync_mode = False
        self.source = self.game
        self.source_active = True
        self.timer.start(self._capture_interval())
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
            self._invalidate()
            self.latest_frame = None
            self.sync_mode = False
            self.source = self.external
            self.source_active = True
            self.timer.start(self._capture_interval())
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
        self._stop_external()
        self._invalidate()
        self.fixed_mode = True
        self.sequence_mode = False
        self.sequence_playing = False
        self.sync_mode = True
        self.timer.stop()
        self.source_stack.setCurrentWidget(self.original_panel)
        self.original_panel.set_static(image, "Image originale")
        self.fixed_original = image
        self.sequence += 1
        self.latest_frame = Frame(image, self.sequence, time.perf_counter())
        self.welcome.hide()
        self.status.setText("Calcul de l'image fixe…")
        self._schedule_workers(self.latest_frame)

    def _show_comparison(self, result):
        request = result.request
        self.last_comparison = result
        self.last_displayed_sequence = request.frame.sequence
        self.displayed_frames = (request.frame.sequence, request.frame.sequence)
        now = time.perf_counter()
        for panel, image, elapsed, network in zip((self.left, self.right), result.images,
                                                   result.milliseconds, result.network_ms):
            panel.set_result(image, elapsed, now, request.size, network, request.frame.sequence)
        if request.fixed:
            self.fixed_first, self.fixed_second = result.images
            self.fixed_size = request.size
            for panel, quality in zip((self.left, self.right), result.metrics):
                panel.stats.setText(panel.stats.text() +
                                    f" | PSNR {quality[0]:.2f} dB | SSIM {quality[1]:.4f}")
            mae, maximum, different = result.difference
            self.status.setText(f"Référence recadrée : {result.reference.shape[1]}×{result.reference.shape[0]} | "
                                f"Écart entre modèles : MAE {mae:.2f}, max {maximum}, {different:.2f}% différents.")
        self._update_metrics()

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
        if self.fixed_first is None or self.fixed_second is None:
            self.status.setText("Chargez d'abord une image fixe.")
            return
        self.showing_difference = not self.showing_difference
        if self.showing_difference:
            self.right.set_static(difference_image(self.fixed_first, self.fixed_second),
                                  f"Différences amplifiées | {self.model_names[1]} - "
                                  f"{self.model_names[0]}")
            self.diff_button.setText(f"Afficher {self.model_names[1]}")
        else:
            self.right.set_static(self.fixed_second, f"{self.model_names[1]} ×{self.scale}")
            self.diff_button.setText("Afficher les différences")

    def _stop_external(self) -> None:
        if self.external:
            self.capture_executor.submit(self.external.close)
            self.external = None

    def _capture_tick(self) -> None:
        if self.closing:
            return
        if self.sequence_mode:
            if self.sequence_reader is not None and self.sequence_index + 1 < len(self.sequence_reader.paths):
                self.sequence_index += 1
                self._show_sequence_frame()
            else:
                self.sequence_playing = False
                self.play_button.setText("Lecture")
                self.timer.stop()
            return
        if self.sync_mode or self.fixed_mode or not self.source_active:
            return
        self._capture_and_schedule()

    def _submit_source(self, function, args, kind, index=None):
        if self.closing or self.capture_future is not None:
            return
        generation = self.generation

        def read():
            return generation, kind, index, function(*args)

        self.capture_future = self.capture_executor.submit(read)
        self.capture_future.add_done_callback(self.source_ready.emit)

    def _source_done(self, future):
        if future is not self.capture_future:
            return
        self.capture_future = None
        if self.closing:
            return
        try:
            generation, kind, index, image = future.result()
        except CancelledError:
            return
        except Exception as exc:
            self.status.setText(f"Capture interrompue : {exc}")
            return
        if generation == self.generation and image is not None:
            if kind == "sequence":
                if index == self.sequence_index:
                    self._sequence_loaded_index = (generation, index)
                    self.original_panel.set_static(image, f"Frame {index + 1}/{len(self.sequence_reader.paths)}")
                    self._accept_capture(image)
            elif not self.sync_mode:
                self._accept_capture(image)
        if self.sequence_mode and (generation != self.generation or index != self.sequence_index):
            self._show_sequence_frame()

    def _capture_and_schedule(self) -> None:
        if self.external is not None:
            self._submit_source(self.external.capture, (), "firefox")
            return
        # Qt impose la capture du widget dans son thread principal.
        try:
            image = self.source.capture()
        except (CaptureError, OSError) as exc:
            self.status.setText(f"Capture interrompue : {exc}")
            return
        if image is not None:
            self._accept_capture(image)

    def _accept_capture(self, image):
        now = time.perf_counter()
        self.sequence += 1
        self.latest_frame = Frame(image, self.sequence, now)
        self.capture_times.append(now)
        self._schedule_workers(self.latest_frame)

    def _schedule_workers(self, frame: Frame) -> None:
        if self.closing:
            return
        self.requests.offer(Request(frame, RESOLUTIONS[self.resolution.currentText()],
                                    self.generation, self.fixed_mode))
        self._dispatch()

    def _update_metrics(self) -> None:
        now = time.perf_counter()
        while self.capture_times and now - self.capture_times[0] > 2:
            self.capture_times.popleft()
        capture_fps = ((len(self.capture_times) - 1) / max(now - self.capture_times[0], 1e-6)
                       if len(self.capture_times) > 1 else 0.0)
        first, second = self.left.display_fps(now), self.right.display_fps(now)
        lot = f" | Paire : {self.last_comparison.total_ms:.1f} ms" if self.last_comparison else ""
        self.metrics.setText(f"Capture réelle : {capture_fps:.1f} FPS | "
                             f"Affichage : {first:.2f} / {second:.2f} FPS | "
                             f"Frame : {self.last_displayed_sequence} | "
                             f"Mode : {'synchronisé' if self.sync_mode else 'temps réel'}{lot}")

    def toggle_sync(self) -> None:
        if self.fixed_mode:
            self.status.setText("Image fixe : ouvrez un jeu ou une séquence pour reprendre la capture.")
            return
        if self.sequence_mode:
            self.toggle_sequence_playback()
            return
        if self.sync_mode:
            self.sync_mode = False
            self.sync_frame = None
            self.pause_button.setText("Pause / Comparer la même image")
            self.status.setText("Mode temps réel actif.")
            self.timer.start(self._capture_interval())
            return
        if self.latest_frame is None:
            self._capture_and_schedule()
        if self.latest_frame is None:
            return
        self.sync_mode = True
        self.sync_frame = self.latest_frame
        self.timer.stop()
        self._schedule_workers(self.sync_frame)
        self.pause_button.setText("Reprendre le temps réel")
        self.status.setText(f"Comparaison synchronisée sur l'image {self.sync_frame.sequence}.")

    def set_fullscreen(self, enabled: bool) -> None:
        self.showFullScreen() if enabled else self.showNormal()

    def toggle_fullscreen(self) -> None:
        self.set_fullscreen(not self.isFullScreen())

    def closeEvent(self, event) -> None:
        if getattr(self, "_shutdown_complete", False):
            event.accept()
            return
        event.ignore()
        if self.closing:
            return
        self.closing = True
        self.status.setText("Libération des modèles…")
        self.timer.stop()
        self.fps_timer.stop()
        self.requests.clear()
        self._stop_external()
        # La libération suit l'inférence en cours ; Qt reste réactif.
        self._release_future = self.executor.submit(self._release_models)
        self._capture_stopped = self.capture_executor.submit(lambda: None)
        self.executor.shutdown(wait=False, cancel_futures=False)
        self.capture_executor.shutdown(wait=False, cancel_futures=False)
        self._close_timer = QTimer(self)
        self._close_timer.timeout.connect(self._finish_close)
        self._close_timer.start(25)

    def _finish_close(self):
        if not self._release_future.done() or not self._capture_stopped.done():
            return
        self._close_timer.stop()
        self.game.close()
        self._shutdown_complete = True
        self.close()


def run() -> int:
    app = QApplication.instance() or QApplication([])
    dialog = ModelSelectionDialog()
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return 0
    first, second, scale = dialog.selection()
    window = MainWindow((first, second), scale)
    window.show()
    return app.exec()
