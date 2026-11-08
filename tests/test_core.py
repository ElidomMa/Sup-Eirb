import unittest
from pathlib import Path
import cv2
import numpy as np

from interface import (CAPTURE_RATES, Frame, RESOLUTIONS, difference_image,
                       image_metrics, pixel_difference_metrics,
                       prepare_fixed_image, resize_for_model)
from sequence import image_paths
from super_resolution import (ModelError, _check_image, available_model_names,
                              correction_checkpoints, model_inventory)


class CoreTests(unittest.TestCase):
    def test_invalid_image_is_rejected(self):
        with self.assertRaises(ModelError):
            _check_image(np.zeros((4, 4), dtype=np.uint8))

    def test_only_models_with_corrections_weights_are_listed(self):
        if correction_checkpoints():
            self.assertEqual(available_model_names(), ("EDSR", "SPAN", "U-Net"))
        else:
            self.assertEqual(available_model_names(), ())

    def test_corrections_inventory_is_explicit(self):
        inventory = model_inventory()
        self.assertEqual(set(inventory), {"EDSR", "SPAN", "Real-ESRGAN", "UNet"})
        self.assertEqual(len(inventory["Real-ESRGAN"]["weights"]),
                         len(list(Path("Existing_Models/Real-ERSGAN/weights").glob("*.pth"))))

    def test_processing_sizes_are_exact_and_aspect_safe(self):
        image = np.zeros((900, 1600, 3), dtype=np.uint8)
        for label, size in RESOLUTIONS.items():
            with self.subTest(label=label):
                result = resize_for_model(image, size)
                self.assertEqual(result.shape[:2], (size[1], size[0]))

    def test_capture_rates_and_frame_identity_are_explicit(self):
        self.assertEqual(CAPTURE_RATES, {"5 FPS": 5, "10 FPS": 10, "15 FPS": 15,
                                         "30 FPS": 30, "60 FPS": 60})
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        first = Frame(image, 1, 10.0)
        second = Frame(image, 2, 10.1)
        self.assertLess(first.captured_at, second.captured_at)
        self.assertNotEqual(first.sequence, second.sequence)

    def test_fixed_image_reconstruction_sizes_and_metrics(self):
        image = np.random.default_rng(3).integers(0, 256, (512, 768, 3), dtype=np.uint8)
        reduced, reference = prepare_fixed_image(image)
        self.assertEqual(reduced.shape, (256, 384, 3))
        restored = cv2.resize(reduced, (768, 512), interpolation=cv2.INTER_CUBIC)
        psnr, ssim = image_metrics(reference, restored)
        self.assertGreater(psnr, 0)
        self.assertGreaterEqual(ssim, -1)
        self.assertLessEqual(ssim, 1)

    def test_difference_image_preserves_dimensions(self):
        first = np.zeros((12, 14, 3), dtype=np.uint8)
        second = np.full_like(first, 20)
        difference = difference_image(first, second)
        self.assertEqual(difference.shape, first.shape)
        self.assertGreater(int(difference.max()), 0)

    def test_pixel_difference_metrics_are_numeric(self):
        first = np.zeros((2, 2, 3), dtype=np.uint8)
        second = first.copy()
        second[0, 0] = (10, 20, 30)
        mae, maximum, percentage = pixel_difference_metrics(first, second)
        self.assertAlmostEqual(mae, 60 / 12)
        self.assertEqual(maximum, 30)
        self.assertAlmostEqual(percentage, 25.0)

    def test_sequence_paths_are_numerically_sorted(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            for name in ("frame_10.png", "frame_2.png", "frame_1.png"):
                Path(directory, name).touch()
            self.assertEqual([path.name for path in image_paths(directory)],
                             ["frame_1.png", "frame_2.png", "frame_10.png"])


if __name__ == "__main__":
    unittest.main()
