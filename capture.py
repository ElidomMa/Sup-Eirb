"""Capture simple d'une fenêtre X11 avec MSS, sans injecter d'événements."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import mss
import numpy as np


@dataclass(frozen=True)
class Window:
    identifier: int
    title: str
    left: int
    top: int
    width: int
    height: int

    @property
    def region(self) -> dict[str, int]:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}


class CaptureError(RuntimeError):
    """Source X11 indisponible ou invalide."""


def list_windows(excluded: set[int] | None = None) -> list[Window]:
    try:
        from Xlib import display
        from Xlib import X
    except ImportError as exc:
        raise CaptureError("python-xlib est nécessaire pour capturer Firefox.") from exc
    excluded = excluded or set()
    try:
        connection = display.Display()
    except Exception as exc:
        raise CaptureError(f"Connexion X11 impossible: {exc}") from exc
    root = connection.screen().root
    result: list[Window] = []
    try:
        for identifier in root.query_tree().children:
            if identifier.id in excluded:
                continue
            geometry = identifier.get_geometry()
            title = identifier.get_wm_name() or ""
            if title and geometry.width > 20 and geometry.height > 20:
                result.append(Window(identifier.id, str(title), geometry.x, geometry.y,
                                     geometry.width, geometry.height))
    finally:
        connection.close()
    return result


class FirefoxCapture:
    """Capture uniquement la région choisie; le jeu reste piloté dans Firefox."""

    def __init__(self, window: Window, region: tuple[int, int, int, int] | None = None) -> None:
        self.window = window
        left, top, width, height = region or (window.left, window.top, window.width, window.height)
        if width < 2 or height < 2:
            raise CaptureError("La région de capture est trop petite.")
        self.area = {"left": left, "top": top, "width": width, "height": height}
        self.screen = mss.mss()

    def capture(self) -> np.ndarray:
        frame = np.asarray(self.screen.grab(self.area), dtype=np.uint8)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def close(self) -> None:
        self.screen.close()
