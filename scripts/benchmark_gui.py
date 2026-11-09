"""Mesure les images réellement peintes par Qt pendant une comparaison."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--baseline-cpu", action="store_true")
parser.add_argument("--seconds", type=float, default=8)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.baseline_cpu:
    sys.path[:] = [path for path in sys.path if path != str(ROOT / ".venv/local-cuda")]

import cv2
import numpy as np
import torch
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


if args.baseline_cpu:
    ui = module("baseline_interface", ROOT / "artifacts/performance/baseline_interface.py")
    adapters = module("baseline_adapters", ROOT / "artifacts/performance/baseline_adapters.py")
    from super_resolution import _weight_path
    classes = {"U-Net": adapters.UNetProcessor, "SPAN": adapters.SPANProcessor}
    ui.create_processor = lambda name, scale: classes[name](scale, _weight_path(name, scale))
else:
    import interface as ui


class TrackedPanel(ui.ImagePanel):
    def __init__(self, *values, **options):
        super().__init__(*values, **options)
        self.painted = []
        self.last_key = None

    def paintEvent(self, event):
        start = time.perf_counter()
        super().paintEvent(event)
        key = self.image.cacheKey()
        if not self.image.isNull() and key != self.last_key:
            self.painted.append((time.perf_counter(), (time.perf_counter() - start) * 1000))
            self.last_key = key


class Source:
    def __init__(self):
        image = cv2.imread(str(ROOT / "test_data/HR/0801.png"))
        self.images = [image, np.ascontiguousarray(image[::-1]), np.ascontiguousarray(image[:, ::-1])]
        self.count = 0

    def capture(self):
        self.count += 1
        return self.images[self.count % len(self.images)]


ui.ImagePanel = TrackedPanel
app = QApplication.instance() or QApplication([])
window = ui.MainWindow(("U-Net", "SPAN"), 2)
window.resize(1100, 800)
window.show()
window.source = Source()
window.source_active = True
beats = []
timer = QTimer()
timer.timeout.connect(lambda: beats.append(time.perf_counter()))
timer.start(10)


def pump(seconds):
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        app.processEvents()
        time.sleep(0.001)


rows = []
for label in ui.RESOLUTIONS:
    window.resolution.setCurrentText(label)
    start = time.perf_counter()
    before = [len(panel.painted) for panel in (window.left, window.right)]
    while any(len(panel.painted) - old < 2 for panel, old in zip((window.left, window.right), before)):
        pump(0.02)
        if time.perf_counter() - start > 90:
            raise RuntimeError("Échauffement Qt trop long : " + window.status.text())
    start = time.perf_counter()
    captured = window.source.count
    pump(args.seconds)
    end = time.perf_counter()
    panels = []
    for panel in (window.left, window.right):
        frames = [(when, cost) for when, cost in panel.painted if start <= when <= end]
        panels.append(dict(frames=len(frames), displayed_fps=len(frames) / (end - start),
                           paint_ms=float(np.mean([cost for _, cost in frames])) if frames else None))
    intervals = np.diff([when for when in beats if start <= when <= end]) * 1000
    row = dict(input=list(ui.RESOLUTIONS[label]), seconds=end - start, panels=panels,
               capture_fps=(window.source.count - captured) / (end - start),
               heartbeat_p95_ms=float(np.percentile(intervals, 95)),
               heartbeat_max_ms=float(max(intervals)))
    rows.append(row)
    print(json.dumps(row), flush=True)
window.close()
if not args.baseline_cpu:
    while not getattr(window, "_shutdown_complete", False):
        pump(0.02)
timer.stop()
args.output.write_text(json.dumps(dict(torch=torch.__version__, models=["U-Net", "SPAN"],
    scale=2, platform=app.platformName(), baseline_cpu=args.baseline_cpu, results=rows), indent=2))
