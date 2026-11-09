"""Adaptateurs simples pour les modèles entraînés de l'équipe."""

from __future__ import annotations

from pathlib import Path
import time

import cv2
import numpy as np

from super_resolution import ModelError, _check_image


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ModelError("PyTorch est requis pour charger les modèles.") from exc
    return torch


def _device(torch):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidualBlockNoBN:
    def __new__(cls, *args, **kwargs):
        torch = _torch()
        nn = torch.nn

        class Block(nn.Module):
            def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
                super().__init__()
                self.res_scale = res_scale
                self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
                self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x):
                return x + self.conv2(self.relu(self.conv1(x))) * self.res_scale

        return Block(*args, **kwargs)


def _edsr_model(torch, scale: int):
    nn = torch.nn
    block = ResidualBlockNoBN

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.img_range = 255.0
            self.mean = torch.Tensor((0.4488, 0.4371, 0.4040)).view(1, 3, 1, 1)
            self.conv_first = nn.Conv2d(3, 64, 3, 1, 1)
            self.body = nn.Sequential(*(block(64, 1, True) for _ in range(16)))
            self.conv_after_body = nn.Conv2d(64, 64, 3, 1, 1)
            layers = []
            if scale in (2, 4):
                for _ in range(1 if scale == 2 else 2):
                    layers.extend((nn.Conv2d(64, 256, 3, 1, 1), nn.PixelShuffle(2)))
            elif scale == 3:
                layers.extend((nn.Conv2d(64, 576, 3, 1, 1), nn.PixelShuffle(3)))
            else:
                raise ModelError(f"Facteur EDSR inconnu: x{scale}.")
            self.upsample = nn.Sequential(*layers)
            self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        def forward(self, x):
            self.mean = self.mean.type_as(x)
            x = (x - self.mean) * self.img_range
            x = self.conv_first(x)
            residual = self.conv_after_body(self.body(x))
            x = self.conv_last(self.upsample(residual + x))
            return x / self.img_range + self.mean

    return Model()


class TorchProcessor:
    def __init__(self, name: str, scale: int, model, weights: Path):
        torch = _torch()
        self.name = name
        self.scale = scale
        self.supported_scales = (scale,)
        self.device = str(_device(torch))
        try:
            state = torch.load(weights, map_location=self.device, weights_only=True)
            if isinstance(state, dict) and "params" in state:
                state = state["params"]
            model.load_state_dict(state, strict=True)
        except Exception as exc:
            raise ModelError(f"Poids incompatibles pour {name} x{scale}: {exc}") from exc
        self.use_half = self.device == "cuda"
        self.model = model.to(self.device).eval()
        if self.use_half:
            self.model.half()
        self.torch = torch

    def upscale(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        _check_image(image)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).unsqueeze(0)
        if self.use_half:
            tensor = tensor.half()
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model(tensor.to(self.device)).clamp(0, 1)
            if self.device == "cuda":
                self.torch.cuda.synchronize()
        result = output[0].mul(255).byte().permute(1, 2, 0).cpu().numpy()
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        expected = (image.shape[0] * self.scale, image.shape[1] * self.scale, 3)
        if result.shape != expected:
            raise ModelError(f"Sortie {self.name} incorrecte: {result.shape}, attendu {expected}.")
        return result, (time.perf_counter() - started) * 1000


class EDSRProcessor(TorchProcessor):
    def __init__(self, scale: int, weights: Path):
        torch = _torch()
        super().__init__("EDSR", scale, _edsr_model(torch, scale), weights)


class SPANProcessor(TorchProcessor):
    def __init__(self, scale: int, weights: Path):
        torch = _torch()
        try:
            from Existing_Models.SPAN.basicsr.archs.span_arch import SPAN
            model = SPAN(3, 3, feature_channels=52, upscale=scale)
        except Exception as exc:
            raise ModelError(f"Architecture SPAN indisponible: {exc}") from exc
        super().__init__("SPAN", scale, model, weights)


class UNetProcessor(TorchProcessor):
    def __init__(self, scale: int, weights: Path):
        torch = _torch()
        try:
            from Existing_Models.UNet.unet import UNetSR
            model = UNetSR(scale=scale, base=32, depth=3)
        except Exception as exc:
            raise ModelError(f"Architecture U-Net indisponible: {exc}") from exc
        super().__init__("U-Net", scale, model, weights)
