"""Navigateur QtWebEngine interactif et capture de son seul contenu."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QUrl, Qt, Signal
from PySide6.QtGui import QImage
from PySide6.QtWebEngineCore import QWebEnginePermission, QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QWidget


class GameView(QWidget):
    status_changed = Signal(str)
    fullscreen_requested = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.browser = QWebEngineView(self)
        settings = self.browser.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.FullScreenSupportEnabled, True)
        self.browser.page().permissionRequested.connect(self._permission)
        self.browser.page().fullScreenRequested.connect(self._fullscreen)
        self.browser.loadStarted.connect(lambda: self.status_changed.emit("Chargement de l'adresse..."))
        self.browser.loadFinished.connect(self._loaded)
        self._closed = False

    def open(self, url: str) -> None:
        self.browser.setUrl(QUrl.fromUserInput(url))

    def _loaded(self, ok: bool) -> None:
        self.status_changed.emit("Jeu prêt: cliquez dans le jeu pour capturer le clavier et la souris."
                                  if ok else "Impossible de charger cette adresse.")

    def _permission(self, permission: QWebEnginePermission) -> None:
        if permission.permissionType() == QWebEnginePermission.PermissionType.MouseLock:
            permission.grant()
        else:
            permission.deny()

    def _fullscreen(self, request) -> None:
        request.accept()
        self.fullscreen_requested.emit(request.toggleOn())

    def capture(self) -> np.ndarray | None:
        if self._closed or not self.isVisible() or self.browser.width() < 2 or self.browser.height() < 2:
            return None
        image = self.browser.grab().toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        if image.isNull():
            return None
        pixels = np.frombuffer(image.bits(), np.uint8).reshape(image.height(), image.bytesPerLine())
        rgba = pixels[:, :image.width() * 4].reshape(image.height(), image.width(), 4)
        return np.ascontiguousarray(rgba[:, :, :3][:, :, ::-1])

    def resizeEvent(self, event) -> None:
        self.browser.setGeometry(self.rect())
        super().resizeEvent(event)

    def close(self) -> None:
        self._closed = True
        self.browser.stop()
        self.browser.deleteLater()
        super().close()
