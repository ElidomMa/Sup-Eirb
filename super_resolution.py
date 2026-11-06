"""Modèles de super-résolution disponibles dans Sup-Eirb."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Protocol

import cv2
import numpy as np


MODEL_PATH = Path(__file__).parent / "models" / "ESPCN_x2.onnx"
UNET_WEIGHTS = Path(__file__).parent / "models" / "unet_x4.pt"


class ModelError(RuntimeError):
    """Erreur explicite de chargement ou d'inférence."""


class Processor(Protocol):
    name: str
    device: str
    supported_scales: tuple[int, ...]

    def upscale(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        ...


def _check_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ModelError("L'image doit être un tableau BGR uint8 de forme (hauteur, largeur, 3).")


def _check_output(image: np.ndarray, result: np.ndarray, scale: int, name: str) -> None:
    expected = (image.shape[0] * scale, image.shape[1] * scale, 3)
    if result.shape != expected:
        raise ModelError(f"Sortie {name} incorrecte: {result.shape}, attendu {expected}.")


class Bicubic:
    """Interpolation OpenCV sans modèle appris."""

    name = "Bicubique"
    device = "CPU"
    supported_scales = (2, 3, 4)

    def __init__(self, scale: int = 2) -> None:
        if scale not in self.supported_scales:
            raise ModelError(f"Facteur Bicubique indisponible: x{scale}.")
        self.scale = scale

    def upscale(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        _check_image(image)
        started = time.perf_counter()
        result = cv2.resize(image, None, fx=self.scale, fy=self.scale,
                            interpolation=cv2.INTER_CUBIC)
        return result, (time.perf_counter() - started) * 1000


class ESPCN:
    """ESPCN x2 exécuté exclusivement avec CUDA."""

    name = "ESPCN"
    device = "CUDA"
    supported_scales = (2,)

    def __init__(self, path: Path = MODEL_PATH) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ModelError("ONNX Runtime GPU est absent.") from exc
        if not path.is_file():
            raise ModelError(f"Modèle ESPCN absent: {path}")
        preload = getattr(ort, "preload_dlls", None)
        if preload is not None:
            preload()
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise ModelError("CUDAExecutionProvider est indisponible; aucun repli CPU ne sera utilisé.")
        try:
            self.session = ort.InferenceSession(
                str(path), providers=["CUDAExecutionProvider"],
                provider_options=[{"cudnn_conv_algo_search": "HEURISTIC"}])
        except Exception as exc:
            raise ModelError(f"Impossible d'initialiser ESPCN avec CUDA: {exc}") from exc
        providers = self.session.get_providers()
        if not providers or providers[0] != "CUDAExecutionProvider":
            raise ModelError(f"ESPCN n'utilise pas CUDA (fournisseurs actifs: {providers}).")
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def upscale(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        _check_image(image)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        tensor = (ycrcb[:, :, 0].astype(np.float32) / 255.0)[None, :, :, None]
        started = time.perf_counter()
        try:
            raw = self.session.run([self.output_name], {self.input_name: tensor})[0]
        except Exception as exc:
            raise ModelError(f"Inférence ESPCN CUDA échouée: {exc}") from exc
        luminance = np.clip(np.rint(raw[0, 0] * 255), 0, 255).astype(np.uint8)
        size = (image.shape[1] * 2, image.shape[0] * 2)
        chroma = cv2.resize(ycrcb[:, :, 1:], size, interpolation=cv2.INTER_LINEAR)
        result = cv2.cvtColor(np.dstack((luminance, chroma)), cv2.COLOR_YCrCb2BGR)
        _check_output(image, result, 2, self.name)
        return result, (time.perf_counter() - started) * 1000


def _unet_available() -> bool:
    return UNET_WEIGHTS.is_file()


def available_model_names() -> tuple[str, ...]:
    names = ["Bicubique"]
    if MODEL_PATH.is_file():
        names.append("ESPCN")
    if _unet_available():
        names.append("U-Net")
    return tuple(names)


def supported_scales(name: str) -> tuple[int, ...]:
    if name == "Bicubique":
        return Bicubic.supported_scales
    if name == "ESPCN":
        return ESPCN.supported_scales
    if name == "U-Net":
        return (4,)
    raise ModelError(f"Modèle inconnu: {name}")


def create_processor(name: str, scale: int) -> Processor:
    if scale not in supported_scales(name):
        raise ModelError(f"{name} ne prend pas en charge x{scale}.")
    if name == "Bicubique":
        return Bicubic(scale)
    if name == "ESPCN":
        return ESPCN()
    if name == "U-Net":
        from unet_adapter import UNetProcessor
        return UNetProcessor(UNET_WEIGHTS)
    raise ModelError(f"Modèle inconnu: {name}")


def create_processors(model_path: Path = MODEL_PATH) -> tuple[Bicubic, ESPCN]:
    """Compatibilité avec l'ancienne API."""
    return Bicubic(), ESPCN(model_path)
