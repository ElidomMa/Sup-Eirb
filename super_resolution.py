"""Détection stricte des modèles fournis dans la branche corrections."""

from __future__ import annotations

from pathlib import Path
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
    return tuple(name for name in ("EDSR", "SPAN", "U-Net")
                 if supported_scales(name))


def supported_scales(name: str) -> tuple[int, ...]:
    weights = {
        "EDSR": (2, 3, 4),
        "SPAN": (2, 3, 4),
        "U-Net": (2, 3, 4),
    }
    return tuple(scale for scale in weights.get(name, ())
                 if _weight_path(name, scale).is_file()
                 and _weight_path(name, scale).stat().st_size > 0)


def create_processor(name: str, scale: int) -> Processor:
    if scale not in supported_scales(name):
        raise ModelError(f"{name} ne possède pas de poids valides pour x{scale}.")
    from model_adapters import EDSRProcessor, SPANProcessor, UNetProcessor
    processors = {"EDSR": EDSRProcessor, "SPAN": SPANProcessor, "U-Net": UNetProcessor}
    try:
        return processors[name](scale, _weight_path(name, scale))
    except KeyError as exc:
        raise ModelError(f"Modèle inconnu: {name}.") from exc


def _weight_path(name: str, scale: int) -> Path:
    paths = {
        "EDSR": CORRECTIONS_ROOT / "EDSR" / "models" / f"edsrx{scale}.pth",
        "SPAN": CORRECTIONS_ROOT / "SPAN" / "models" / f"spanx{scale}_ch52.pth",
        "U-Net": CORRECTIONS_ROOT / "UNet" / "checkpoints" / f"x{scale}" /
        f"unet_x{scale}_best.pt",
    }
    return paths.get(name, Path(""))


def model_inventory() -> dict[str, dict[str, object]]:
    """Décrit les familles et les poids présents localement."""
    return {
        "EDSR": {"architecture": "EDSR", "scales": (2, 3, 4),
                 "weights": tuple(_weight_path("EDSR", s).name for s in (2, 3, 4)
                                  if _weight_path("EDSR", s).is_file())},
        "SPAN": {"architecture": "SPAN", "scales": (2, 3, 4),
                 "weights": tuple(_weight_path("SPAN", s).name for s in (2, 3, 4)
                                  if _weight_path("SPAN", s).is_file())},
        "Real-ESRGAN": {"architecture": "Real-ESRGAN", "scales": (2, 4),
                        "weights": tuple(path.name for path in sorted(
                            (CORRECTIONS_ROOT / "Real-ERSGAN" / "weights").glob("*.pth")
                        ))},
        "UNet": {"architecture": "UNet", "scales": (2, 3, 4),
                 "weights": tuple(_weight_path("U-Net", s).name for s in (2, 3, 4)
                                  if _weight_path("U-Net", s).is_file())},
    }
