"""Chargement borné et historique des séquences d'images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import cv2
import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def _natural_key(path: Path) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part.lower()
                 for part in re.split(r"(\d+)", path.name))


def image_paths(folder: str | Path) -> list[Path]:
    """Retourne les images dans un ordre numérique stable."""
    root = Path(folder)
    return sorted((path for path in root.iterdir()
                   if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
                  key=_natural_key)


@dataclass
class SequenceResult:
    frame: int
    filename: str
    result_1: np.ndarray
    result_2: np.ndarray
    time_1_ms: float
    time_2_ms: float
    psnr_1: float | None = None
    psnr_2: float | None = None
    ssim_1: float | None = None
    ssim_2: float | None = None


class SequenceReader:
    """Ne garde qu'une image source à la fois en mémoire."""

    def __init__(self, folder: str | Path) -> None:
        self.paths = image_paths(folder)
        if not self.paths:
            raise ValueError(f"Aucune image PNG/JPEG trouvée dans {folder}.")

    def read(self, index: int) -> np.ndarray:
        if not 0 <= index < len(self.paths):
            raise IndexError(index)
        image = cv2.imread(str(self.paths[index]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Image illisible: {self.paths[index]}")
        return image


class ResultHistory:
    """Historique limité; les séquences longues ne saturent pas la RAM."""

    def __init__(self, limit: int = 120) -> None:
        self.limit = limit
        self._items: dict[int, SequenceResult] = {}

    def add(self, item: SequenceResult) -> None:
        self._items[item.frame] = item
        for frame in sorted(self._items)[:-self.limit]:
            del self._items[frame]

    def get(self, frame: int) -> SequenceResult | None:
        return self._items.get(frame)

    def __iter__(self) -> Iterable[SequenceResult]:
        return iter(self._items.values())
