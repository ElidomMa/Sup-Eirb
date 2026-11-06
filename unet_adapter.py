"""Adaptateur Sup-Eirb pour le U-Net original de l'équipe."""

from __future__ import annotations

from pathlib import Path
import time

import cv2
import numpy as np

from super_resolution import ModelError


class UNetProcessor:
    """Charge les poids fournis sans modifier `Existing_Models/UNet`."""

    name = "U-Net"
    device = "CUDA"
    supported_scales = (4,)

    def __init__(self, weights: Path, base: int = 32, depth: int = 3) -> None:
        try:
            import torch
            from Existing_Models.UNet.unet import UNetSR
        except ImportError as exc:
            raise ModelError("PyTorch est requis pour le U-Net.") from exc
        if not weights.is_file():
            raise ModelError(f"Poids U-Net absents: {weights}")
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNetSR(scale=4, base=base, depth=depth).to(self.device)
        try:
            state = torch.load(weights, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
        except Exception as exc:
            raise ModelError(f"Impossible de charger les poids U-Net: {exc}") from exc
        self.model.eval()

    def upscale(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ModelError("L'image doit être un tableau BGR uint8 de forme (hauteur, largeur, 3).")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).div(255)
        started = time.perf_counter()
        with self.torch.inference_mode():
            result = self.model(tensor.to(self.device)).clamp(0, 1)
            if self.device.type == "cuda":
                self.torch.cuda.synchronize()
        output = result[0].mul(255).byte().permute(1, 2, 0).cpu().numpy()
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        expected = (image.shape[0] * 4, image.shape[1] * 4, 3)
        if output.shape != expected:
            raise ModelError(f"Sortie U-Net incorrecte: {output.shape}, attendu {expected}.")
        return output, (time.perf_counter() - started) * 1000
