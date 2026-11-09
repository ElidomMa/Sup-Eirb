"""Vérification Qt : paires, saturation, navigation et libération mémoire."""

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QDialog

from interface import MainWindow


class SlowProcessor:
    device = "cpu"
    use_half = False

    def __init__(self, name, scale, calls):
        self.name, self.scale, self.calls = name, scale, calls

    def upscale(self, image):
        self.calls.append((self.name, id(image), image.shape))
        time.sleep(0.06)
        return cv2.resize(image, None, fx=self.scale, fy=self.scale), 60.0

    def close(self):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    app = QApplication.instance() or QApplication([])
    heartbeats = []
    heartbeat = QTimer()
    heartbeat.timeout.connect(lambda: heartbeats.append(time.perf_counter()))
    heartbeat.start(10)

    def wait(condition, timeout=60):
        deadline = time.perf_counter() + timeout
        while not condition():
            app.processEvents()
            if time.perf_counter() > deadline:
                raise AssertionError("Délai dépassé : " + window.status.text())
            time.sleep(0.002)
        app.processEvents()

    calls = []
    factory = (None if args.real else patch("interface.create_processor",
               side_effect=lambda name, scale: SlowProcessor(name, scale, calls)))
    if factory:
        factory.start()
    window = MainWindow(("U-Net", "SPAN"), 2)
    window.show()
    wait(lambda: window.processors is not None and window.future is None)
    window.timer.stop()
    image = cv2.imread("test_data/HR/0801.png")
    assert image is not None
    window.sync_mode = True
    window._accept_capture(image)
    wait(lambda: window.last_displayed_sequence == window.sequence)
    assert window.displayed_frames == (window.sequence, window.sequence)
    initial = len(calls)
    for _ in range(5):
        window._schedule_workers(window.latest_frame)
    assert window.future is None and len(calls) == initial

    # Le GPU reste occupé pendant que Qt reçoit de nouvelles captures.
    for _ in range(20):
        window._accept_capture(image)
    target = window.sequence
    wait(lambda: window.last_displayed_sequence == target)
    assert window.displayed_frames == (target, target)
    if not args.real:
        assert len(calls) - initial <= 4, calls
        for index in range(0, len(calls), 2):
            assert calls[index][1] == calls[index + 1][1]

    for label, size in (("480 × 270", (480, 270)), ("640 × 360", (640, 360))):
        window.resolution.setCurrentText(label)
        wait(lambda: window.last_comparison is not None and window.future is None)
        assert window.last_comparison.request.size == size
        assert window.left.pixels.shape == (size[1] * 2, size[0] * 2, 3)
    window.change_zoom(2)
    window.resize(700, 500)
    window.toggle_expanded()
    app.processEvents()
    window.reset_zoom()
    window.toggle_expanded()

    with patch("interface.QFileDialog.getOpenFileName", return_value=("test_data/HR/0801.png", "")):
        window.open_image()
    wait(lambda: window.last_comparison is not None and window.last_comparison.request.fixed)
    assert "PSNR" in window.left.stats.text() and "SSIM" in window.right.stats.text()
    window.toggle_difference()
    window.toggle_difference()
    assert window.fixed_first.shape == (720, 1280, 3)

    with tempfile.TemporaryDirectory() as folder:
        for i in range(3):
            cv2.imwrite(str(Path(folder) / f"frame_{i}.png"), image[::4, ::4])
        with patch("interface.QFileDialog.getExistingDirectory", return_value=folder):
            window.open_sequence()
        wait(lambda: window.last_comparison is not None and window.future is None)
        window.next_sequence_frame()
        wait(lambda: window.last_displayed_sequence == window.sequence and window.capture_future is None and window.future is None)
        assert window.sequence_index == 1
        window.previous_sequence_frame()
        wait(lambda: window.capture_future is None and window.future is None)
        assert window.sequence_index == 0
        window.toggle_sequence_playback()
        wait(lambda: not window.sequence_playing and window.capture_future is None and window.future is None)
        assert window.sequence_index == 2
        window.restart_sequence()
        wait(lambda: window.capture_future is None and window.future is None)
        assert window.sequence_index == 0

    # Le navigateur reste le même lors du changement des modèles.
    browser = window.game.browser
    with patch("interface.ModelSelectionDialog") as dialog:
        dialog.return_value.exec.return_value = QDialog.DialogCode.Accepted
        dialog.return_value.selection.return_value = ("EDSR", "U-Net", 3)
        window.change_models()
    wait(lambda: window.model_change is None and window.future is None)
    assert window.game.browser is browser
    assert window.scale == 3
    assert window.left.pixels.shape == (1080, 1920, 3)

    # Capture locale du navigateur, sans dépendre d'un site extérieur.
    window.sequence_mode = window.fixed_mode = False
    window.sync_mode = True
    window.source_stack.setCurrentWidget(window.game)
    loaded = []
    browser.loadFinished.connect(loaded.append)
    browser.setHtml("<html><body style='background:#246;color:white'>Sup-Eirb — test capture</body></html>")
    wait(lambda: bool(loaded))
    captured = window.game.capture()
    assert captured is not None and captured.ndim == 3
    window.grab().save(str(args.output.with_suffix(".png")))

    before_close = len(heartbeats)
    window._accept_capture(image)
    window.close()
    wait(lambda: getattr(window, "_shutdown_complete", False))
    assert window.processors is None
    assert len(heartbeats) > before_close
    heartbeat.stop()
    report = dict(real=args.real, qt_platform=app.platformName(), checks="passed",
                  displayed_same_frame=True, model_calls=len(calls),
                  heartbeat_max_ms=max(np.diff(heartbeats), default=0) * 1000,
                  heartbeat_p95_ms=float(np.percentile(np.diff(heartbeats), 95) * 1000),
                  browser_capture_shape=list(captured.shape))
    if args.real:
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        report["allocated_after_close_mib"] = torch.cuda.memory_allocated() / 2**20
        report["reserved_after_close_mib"] = torch.cuda.memory_reserved() / 2**20
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if factory:
        factory.stop()


if __name__ == "__main__":
    main()
