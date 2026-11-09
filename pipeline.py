"""Un lot contient toujours les deux résultats de la même image."""

from dataclasses import dataclass
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    sequence: int
    captured_at: float


@dataclass(frozen=True)
class Request:
    frame: Frame
    size: tuple[int, int]
    generation: int
    fixed: bool = False


@dataclass
class Comparison:
    request: Request
    images: tuple[np.ndarray, np.ndarray]
    milliseconds: tuple[float, float]
    network_ms: tuple[float, float]
    total_ms: float
    reference: np.ndarray | None = None
    metrics: tuple | None = None
    difference: tuple | None = None


def resize_for_model(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Ajustement puis recadrage central, sans déformation."""
    target_width, target_height = size
    height, width = image.shape[:2]
    if (width, height) == size:
        return np.ascontiguousarray(image)
    scale = max(target_width / width, target_height / height)
    resized = cv2.resize(image, (round(width * scale), round(height * scale)),
                         interpolation=cv2.INTER_AREA)
    top = (resized.shape[0] - target_height) // 2
    left = (resized.shape[1] - target_width) // 2
    return np.ascontiguousarray(resized[top:top + target_height, left:left + target_width])


def process_pair(processors, request: Request) -> Comparison:
    started = time.perf_counter()
    first, second = processors
    reference = None
    source = request.frame.image
    if request.fixed:
        reference = resize_for_model(source, tuple(n * first.scale for n in request.size))
        source = reference
    low = resize_for_model(source, request.size)
    # Une seule conversion et un seul transfert pour la paire compatible.
    shared = (hasattr(first, "prepare_input") and hasattr(second, "infer_prepared")
              and first.device == second.device and first.dtype == second.dtype)
    tensor = first.prepare_input(low) if shared else None
    images, times, network = [], [], []
    for processor in processors:
        image, elapsed = (processor.infer_prepared(tensor, low.shape) if shared
                          else processor.upscale(low))
        images.append(image)
        times.append(elapsed)
        network.append(getattr(processor, "last_timings", {}).get("network_ms", elapsed))
    return Comparison(request, tuple(images), tuple(times), tuple(network),
                      (time.perf_counter() - started) * 1000, reference)


class LatestRequest:
    """Une seule image en attente ; les captures anciennes sont remplacées."""

    def __init__(self):
        self.pending = None
        self.last_key = None

    @staticmethod
    def key(request):
        return request.generation, request.frame.sequence, request.size, request.fixed

    def offer(self, request):
        if self.key(request) != self.last_key:
            self.pending = request

    def take(self):
        request, self.pending = self.pending, None
        if request is not None:
            self.last_key = self.key(request)
        return request

    def clear(self):
        self.pending = None
        self.last_key = None
