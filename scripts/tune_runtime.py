"""Compare précision, disposition mémoire et concurrence sur le GPU local."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import itertools
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from benchmark import source_images
from super_resolution import available_model_names, create_processor, supported_scales


def measured(function, runs=6):
    for _ in range(2):
        function()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        function()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return dict(ms=float(np.mean(samples)), samples=samples,
                allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                reserved_mib=torch.cuda.max_memory_reserved() / 2**20)


def tune():
    image = source_images((480, 270))[0]
    rows = []
    for name in available_model_names():
        for precision, channels_last, autotune in itertools.product(("fp32", "fp16"), (False, True), (False, True)):
            torch.backends.cudnn.benchmark = autotune
            if hasattr(torch.backends.cudnn, "benchmark_limit"):
                torch.backends.cudnn.benchmark_limit = 4
            processor = create_processor(name, 2, device="cuda:0", precision=precision)
            layout = torch.channels_last if channels_last else torch.contiguous_format
            processor.model.to(memory_format=layout)
            tensor = processor.prepare_input(image).contiguous(memory_format=layout)
            row = dict(model=name, precision=precision, channels_last=channels_last, autotune=autotune)
            row.update(measured(lambda: processor.infer_prepared(tensor, image.shape)))
            print(json.dumps(row), flush=True)
            rows.append(row)
            del tensor
            processor.close()
    return rows


def quality():
    rows = []
    for name in available_model_names():
        for scale in supported_scales(name):
            reference = create_processor(name, scale, device="cuda:0", precision="fp32", optimize=False)
            processor = create_processor(name, scale, device="cuda:0", precision="auto")
            for size in ((320, 180), (480, 270), (640, 360)):
                for index, image in enumerate(source_images(size)):
                    original, _ = reference.upscale(image)
                    result, _ = processor.upscale(image)
                    delta = result.astype(np.float32) - original.astype(np.float32)
                    mse = float(np.mean(delta**2))
                    with torch.inference_mode():
                        tensor = processor.prepare_input(image)
                        output = processor.model(tensor)
                        finite = bool(torch.isfinite(output).all())
                        del tensor, output
                    row = dict(model=name, scale=scale, input=list(size), image=index,
                               device=processor.verify_device(), input_device=processor.last_input_device,
                               output_device=processor.last_output_device,
                               mae=float(np.mean(np.abs(delta))), max_error=float(np.max(np.abs(delta))),
                               psnr=None if mse == 0 else float(10 * np.log10(255**2 / mse)),
                               finite=finite, output_shape=list(result.shape),
                               output_std=float(result.std()))
                    assert finite and result.shape == original.shape
                    assert row["mae"] < 0.5 and row["max_error"] <= 4, row
                    rows.append(row)
                    print(json.dumps(row), flush=True)
            reference.close()
            processor.close()
    gc.collect()
    torch.cuda.empty_cache()
    return dict(results=rows, allocated_after_release_mib=torch.cuda.memory_allocated() / 2**20)


def pairs():
    rows = []
    for scale in (2, 3, 4):
        for names in itertools.combinations(available_model_names(), 2):
            processors = [create_processor(name, scale, device="cuda:0") for name in names]
            for size in ((320, 180), (480, 270), (640, 360)):
                images = source_images(size)
                for concurrent in (False, True):
                    count = 0
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        def execute():
                            nonlocal count
                            image = images[count % len(images)]
                            count += 1
                            if concurrent:
                                jobs = [pool.submit(p.upscale, image) for p in processors]
                                return [job.result() for job in jobs]
                            return [p.upscale(image) for p in processors]
                        row = dict(models=list(names), scale=scale, input=list(size), concurrent=concurrent)
                        row.update(measured(execute))
                        rows.append(row)
                        print(json.dumps(row), flush=True)
            for processor in processors:
                processor.close()
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("tune", "quality", "pairs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"tune": tune, "quality": quality, "pairs": pairs}[args.mode]()
    args.output.write_text(json.dumps(result, indent=2))
