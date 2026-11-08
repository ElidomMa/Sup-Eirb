"""Détection stricte des modèles fournis dans la branche corrections."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Protocol

import numpy as np


CORRECTIONS_ROOT = Path(__file__).parent / "Existing_Models"
CHECKPOINT_SUFFIXES = {".pth", ".pt", ".onnx", ".ckpt", ".safetensors"}


class ModelError(RuntimeError):
    """Erreur explicite de chargement ou d'inférence."""


class Processor(Protocol):
    name: str
    device: str
    supported_scales: tuple[int, ...]

    def upscale(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        ...


def _check_image(image: np.ndarray) -> None:
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3):
        raise ModelError("L'image doit être un tableau BGR uint8 de forme (hauteur, largeur, 3).")


def correction_checkpoints() -> tuple[Path, ...]:
    """Retourne les poids réellement présents dans les modèles de corrections."""
    if not CORRECTIONS_ROOT.is_dir():
        return ()
    return tuple(sorted(
        path for path in CORRECTIONS_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in CHECKPOINT_SUFFIXES
    ))


def available_model_names() -> tuple[str, ...]:
    """Ne propose aucun modèle sans adaptateur et poids vérifiables."""
    # La branche corrections actuellement récupérée ne contient aucun checkpoint.
    # Un modèle n'est ajouté ici qu'après intégration d'un adaptateur testable.
    return ()


def supported_scales(name: str) -> tuple[int, ...]:
    raise ModelError(f"Modèle de corrections indisponible: {name}.")


def create_processor(name: str, scale: int) -> Processor:
    raise ModelError(
        f"Le modèle {name} ne peut pas être chargé: aucun poids entraîné "
        "et adaptateur validé provenant de corrections."
    )


def model_inventory() -> dict[str, dict[str, object]]:
    """Décrit les familles présentes dans corrections, sans les déclarer utilisables."""
    return {
        "EDSR": {"architecture": "EDSR", "scales": (2, 3, 4), "weights": ()},
        "SPAN": {"architecture": "SPAN", "scales": (2, 3, 4), "weights": ()},
        "Real-ESRGAN": {"architecture": "Real-ESRGAN", "scales": (4,), "weights": ()},
        "UNet": {"architecture": "UNet", "scales": (2, 3, 4), "weights": ()},
    }


def inference_time(started: float) -> float:
    """Convertit une mesure monotone en millisecondes."""
    return (time.perf_counter() - started) * 1000
