"""Benchmark reproductible des vrais poids, sans téléchargement."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import torch
from PySide6.QtWidgets import QApplication

from interface import ImagePanel, resize_for_model
from super_resolution import available_model_names, create_processor, supported_scales, _weight_path


def source_images(size):
    paths = sorted((ROOT / "test_data" / "HR").glob("*.png"))
    if not paths:
        raise RuntimeError("Ajoutez une image PNG dans test_data/HR pour le benchmark.")
    source = cv2.imread(str(paths[0]))
    h, w = source.shape[:2]
    return [resize_for_model(crop, size) for crop in
            (source, source[:h * 3 // 4, :w * 3 // 4], source[h // 4:, w // 4:])]


def original_upscale(processor, image):
    """Même calcul que l'adaptateur initial, avec des frontières mesurées."""
    cuda = processor.device.startswith("cuda")

    def stamp():
        if cuda:
            torch.cuda.synchronize()
        return time.perf_counter()

    marks = [stamp()]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).unsqueeze(0)
    if processor.use_half:
        tensor = tensor.half()
    marks.append(stamp())
    tensor = tensor.to(processor.device)
    marks.append(stamp())
    with torch.inference_mode():
        output = processor.model(tensor).clamp(0, 1)
    marks.append(stamp())
    packed = output[0].mul(255).byte().permute(1, 2, 0)
    packed_at = stamp()
    result = packed.cpu().numpy()
    copied_at = stamp()
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    end = stamp()
    return result, dict(preprocess_ms=(marks[1] - marks[0]) * 1000,
                        h2d_ms=(marks[2] - marks[1]) * 1000,
                        network_ms=(marks[3] - marks[2]) * 1000,
                        d2h_ms=(copied_at - packed_at) * 1000,
                        postprocess_ms=(packed_at - marks[3] + end - copied_at) * 1000,
                        total_ms=(end - marks[0]) * 1000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(available_model_names()))
    parser.add_argument("--scales", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--widths", nargs="+", type=int, default=[320, 480, 640])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument("--precision", choices=["fp32", "fp16", "auto"], default="auto")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.runs < 3 or args.warmup < 1:
        parser.error("Au moins trois mesures et un échauffement sont requis.")
    legacy = None
    if args.baseline:
        spec = importlib.util.spec_from_file_location("baseline_adapters", args.baseline)
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        if args.device != "auto":
            legacy._device = lambda _: torch.device(args.device)
    app = QApplication.instance() or QApplication([])
    panel = ImagePanel("Benchmark")
    panel.resize(640, 360)
    panel.show()
    app.processEvents()
    report = dict(torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  threads=torch.get_num_threads(), warmup=args.warmup, runs=args.runs,
                  source="Trois cadrages de test_data/HR/0801.png",
                  qt_platform=app.platformName(), baseline=bool(legacy), results=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        for scale in args.scales:
            if scale not in supported_scales(name):
                continue
            if legacy:
                cls = {"EDSR": legacy.EDSRProcessor, "SPAN": legacy.SPANProcessor,
                       "U-Net": legacy.UNetProcessor}[name]
                processor = cls(scale, _weight_path(name, scale))
                if args.precision != "auto":
                    processor.use_half = args.precision == "fp16"
                    processor.model.to(dtype=torch.float16 if processor.use_half else torch.float32)
            else:
                processor = create_processor(name, scale, device=args.device, precision=args.precision)
            for width in args.widths:
                size = (width, width * 9 // 16)
                images = source_images(size)
                row = dict(model=name, scale=scale, input=list(size), device=processor.device,
                           precision="fp16" if processor.use_half else "fp32")
                try:
                    run = (lambda image: original_upscale(processor, image)) if legacy else processor.profile
                    for i in range(args.warmup):
                        run(images[i % len(images)])
                    if processor.device.startswith("cuda"):
                        torch.cuda.reset_peak_memory_stats()
                    samples = []
                    for i in range(args.runs):
                        result, timing = run(images[i % len(images)])
                        assert result.shape == (size[1] * scale, size[0] * scale, 3)
                        started = time.perf_counter()
                        panel.set_result(result, timing["total_ms"], started, size)
                        app.processEvents()
                        timing["qt_ms"] = (time.perf_counter() - started) * 1000
                        samples.append(timing)
                    row.update({key: float(np.mean([s[key] for s in samples])) for key in samples[0]})
                    row["network_fps"] = 1000 / row["network_ms"]
                    row["pipeline_fps"] = 1000 / row["total_ms"]
                    row["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20 if processor.device.startswith("cuda") else 0
                    row["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20 if processor.device.startswith("cuda") else 0
                    row["samples"] = samples
                except (RuntimeError, AssertionError) as exc:
                    row["error"] = str(exc)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                report["results"].append(row)
                args.output.write_text(json.dumps(report, indent=2))
                print(json.dumps({k: v for k, v in row.items() if k != "samples"}), flush=True)
            if hasattr(processor, "close"):
                processor.close()
            del processor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    panel.close()


if __name__ == "__main__":
    main()
