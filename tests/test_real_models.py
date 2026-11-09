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
                    self.assertEqual(processor.verify_device(), processor.device)
                    self.assertEqual(processor.last_input_device, processor.device)
                    self.assertEqual(processor.last_output_device, processor.device)
                    self.assertFalse(processor.model.training)
                    self.assertTrue(all(not p.requires_grad for p in processor.model.parameters()))
                    processor.close()

    def test_span_fusion_is_cached_without_changing_the_output(self):
        image = np.random.default_rng(7).integers(0, 256, (24, 32, 3), dtype=np.uint8)
        original = create_processor("SPAN", 2, device="cpu", precision="fp32", optimize=False)
        cached = create_processor("SPAN", 2, device="cpu", precision="fp32")
        expected, _ = original.upscale(image)
        result, _ = cached.upscale(image)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(cached.fused_blocks, 20)
        # Deuxième image : aucune nouvelle fusion, même résultat.
        np.testing.assert_array_equal(cached.upscale(image)[0], expected)
        original.close()
        cached.close()

    def test_two_models_have_compatible_scales(self):
        names = available_model_names()
        self.assertGreaterEqual(len(names), 2)
        common = set(supported_scales(names[0]))
        common.intersection_update(*(supported_scales(name) for name in names[1:]))
        self.assertEqual(common, {2, 3, 4})


if __name__ == "__main__":
    unittest.main()
