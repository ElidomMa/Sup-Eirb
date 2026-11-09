"""Adaptateurs simples pour les modèles entraînés de l'équipe."""

from __future__ import annotations

from pathlib import Path
import time
import warnings

import cv2
import numpy as np

from super_resolution import ModelError, _check_image


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ModelError("PyTorch est requis pour charger les modèles.") from exc
    return torch


def _device(torch, requested="auto"):
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        reason = "PyTorch installé sans CUDA" if torch.version.cuda is None else "pilote CUDA inaccessible"
        warnings.warn(f"Exécution CPU : {reason}. Les modèles seront lents.", RuntimeWarning)
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ModelError("CUDA demandé mais indisponible dans cet environnement Python.")
        return torch.device("cuda", device.index or 0)
    if device.type != "cpu":
        raise ModelError(f"Périphérique non pris en charge : {device}.")
    return device


def _cached_params():
    """Les poids sont figés : la fusion SPAN a déjà été calculée."""


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
    def __init__(self, name: str, scale: int, model, weights: Path, *,
                 device="auto", precision="auto", optimize=True):
        torch = _torch()
        self.name = name
        self.scale = scale
        self.supported_scales = (scale,)
        self.device = str(_device(torch, device))
        self.is_cuda = self.device.startswith("cuda")
        if precision not in ("auto", "fp16", "fp32"):
            raise ModelError(f"Précision inconnue : {precision}.")
        if precision == "fp16" and not self.is_cuda:
            raise ModelError("FP16 n'est proposé que sur CUDA.")
        try:
            # Chargement CPU : pas de seconde copie des poids sur le GPU.
            state = torch.load(weights, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "params" in state:
                state = state["params"]
            model.load_state_dict(state, strict=True)
        except Exception as exc:
            raise ModelError(f"Poids incompatibles pour {name} x{scale}: {exc}") from exc
        del state
        model.eval().requires_grad_(False)
        self.fused_blocks = 0
        if optimize and name == "SPAN":
            # Utilise la fusion prévue par l'architecture, une seule fois en FP32.
            for block in model.modules():
                if hasattr(block, "update_params"):
                    block.update_params()
                    block.update_params = _cached_params
                    block.weight_concat = block.bias_concat = None
                    self.fused_blocks += 1
        # GTX 1650 mesurée : FP32 est plus rapide que FP16 pour ces trois réseaux.
        self.use_half = precision == "fp16"
        self.dtype = torch.float16 if self.use_half else torch.float32
        self.model = model.to(device=self.device, dtype=self.dtype)
        if hasattr(self.model, "mean"):
            self.model.mean = self.model.mean.to(device=self.device, dtype=self.dtype)
        self.torch = torch
        self.last_timings = {}
        self.last_input_device = self.last_output_device = None
        self.verify_device()

    def verify_device(self):
        expected = self.torch.device(self.device)
        tensors = list(self.model.parameters()) + list(self.model.buffers())
        if not tensors or any(tensor.device != expected for tensor in tensors):
            raise ModelError(f"{self.name} : paramètres ou buffers hors de {expected}.")
        return str(tensors[0].device)

    def prepare_input(self, image):
        _check_image(image)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        # Transfert uint8, conversion et normalisation sur le périphérique cible.
        return tensor.to(device=self.device).to(dtype=self.dtype).div_(255)

    def infer_prepared(self, tensor, input_shape):
        if self.model is None:
            raise ModelError(f"{self.name} a été libéré.")
        if tensor.device != self.torch.device(self.device) or tensor.dtype != self.dtype:
            raise ModelError(f"Entrée {self.name} incompatible avec {self.device}/{self.dtype}.")
        self.last_input_device = str(tensor.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            if self.is_cuda:
                begin, end = self.torch.cuda.Event(enable_timing=True), self.torch.cuda.Event(enable_timing=True)
                begin.record()
            output = self.model(tensor).clamp_(0, 1)
            if self.is_cuda:
                end.record()
            network_ms = (time.perf_counter() - started) * 1000
            self.last_output_device = str(output.device)
            packed = output[0].mul_(255).to(self.torch.uint8).permute(1, 2, 0).contiguous()
            result = cv2.cvtColor(packed.cpu().numpy(), cv2.COLOR_RGB2BGR)
            if self.is_cuda:
                network_ms = begin.elapsed_time(end)
        self._check_output(result, input_shape)
        total = (time.perf_counter() - started) * 1000
        self.last_timings = {"network_ms": network_ms, "total_ms": total}
        return result, total

    def _check_output(self, result, shape):
        expected = (shape[0] * self.scale, shape[1] * self.scale, 3)
        if result.shape != expected:
            raise ModelError(f"Sortie {self.name} incorrecte: {result.shape}, attendu {expected}.")

    def upscale(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        _check_image(image)
        started = time.perf_counter()
        result, _ = self.infer_prepared(self.prepare_input(image), image.shape)
        elapsed = (time.perf_counter() - started) * 1000
        self.last_timings["total_ms"] = elapsed
        return result, elapsed

    def profile(self, image):
        """Mesures détaillées avec synchronisation aux frontières CUDA."""
        _check_image(image)

        def stamp():
            if self.is_cuda:
                self.torch.cuda.synchronize(self.device)
            return time.perf_counter()

        start = stamp()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        pre = stamp()
        tensor = tensor.to(self.device).to(self.dtype).div_(255)
        transferred = stamp()
        self.last_input_device = str(tensor.device)
        with self.torch.inference_mode():
            output = self.model(tensor).clamp_(0, 1)
            network = stamp()
            self.last_output_device = str(output.device)
            packed = output[0].mul_(255).to(self.torch.uint8).permute(1, 2, 0).contiguous()
            packed_at = stamp()
            result = packed.cpu().numpy()
            copied_at = stamp()
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        end = stamp()
        self._check_output(result, image.shape)
        timing = dict(preprocess_ms=(pre - start) * 1000,
                      h2d_ms=(transferred - pre) * 1000,
                      network_ms=(network - transferred) * 1000,
                      d2h_ms=(copied_at - packed_at) * 1000,
                      postprocess_ms=(packed_at - network + end - copied_at) * 1000,
                      total_ms=(end - start) * 1000)
        self.last_timings = timing
        return result, timing

    def close(self):
        self.model = None
        if self.is_cuda:
            self.torch.cuda.empty_cache()


class EDSRProcessor(TorchProcessor):
    def __init__(self, scale: int, weights: Path, **options):
        torch = _torch()
        super().__init__("EDSR", scale, _edsr_model(torch, scale), weights, **options)


class SPANProcessor(TorchProcessor):
    def __init__(self, scale: int, weights: Path, **options):
        torch = _torch()
        try:
            from Existing_Models.SPAN.basicsr.archs.span_arch import SPAN
            model = SPAN(3, 3, feature_channels=52, upscale=scale)
        except Exception as exc:
            raise ModelError(f"Architecture SPAN indisponible: {exc}") from exc
        super().__init__("SPAN", scale, model, weights, **options)


class UNetProcessor(TorchProcessor):
    def __init__(self, scale: int, weights: Path, **options):
        torch = _torch()
        try:
            from Existing_Models.UNet.unet import UNetSR
            model = UNetSR(scale=scale, base=32, depth=3)
        except Exception as exc:
            raise ModelError(f"Architecture U-Net indisponible: {exc}") from exc
        super().__init__("U-Net", scale, model, weights, **options)
