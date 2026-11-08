import unittest

import numpy as np

from super_resolution import available_model_names, create_processor, supported_scales


class RealModelTests(unittest.TestCase):
    """Inférences courtes sur les poids locaux fournis par l'équipe."""

    @unittest.skipUnless(available_model_names(), "poids locaux absents")
    def test_each_available_model_runs_inference(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        for name in available_model_names():
            for scale in supported_scales(name):
                with self.subTest(model=name, scale=scale):
                    processor = create_processor(name, scale)
                    result, elapsed = processor.upscale(image)
                    self.assertEqual(result.shape, (8 * scale, 8 * scale, 3))
                    self.assertGreaterEqual(elapsed, 0)

    def test_two_models_have_compatible_scales(self):
        names = available_model_names()
        self.assertGreaterEqual(len(names), 2)
        common = set(supported_scales(names[0]))
        common.intersection_update(*(supported_scales(name) for name in names[1:]))
        self.assertEqual(common, {2, 3, 4})


if __name__ == "__main__":
    unittest.main()
