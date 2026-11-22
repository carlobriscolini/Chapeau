import unittest

import numpy as np

from research.mask_variants import cropped_shape, perturb_minority
from research.synth import bbox, is_rectangle


class MaskVariantsTest(unittest.TestCase):
    def test_seeded_rectangle_variants_change_dimensions_and_position(self):
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[112:192, 230:350] = 1
        source = mask.copy()
        variants = [perturb_minority(mask, np.random.default_rng(seed)) for seed in range(12)]
        np.testing.assert_array_equal(variants[0], perturb_minority(mask, np.random.default_rng(0)))
        np.testing.assert_array_equal(mask, source)

        aspects = set()
        positions = set()
        for variant in variants:
            self.assert_binary_with_similar_area(variant, mask)
            self.assertTrue(is_rectangle(variant))
            self.assertNotEqual(cropped_shape(variant), cropped_shape(mask))
            y0, y1, x0, x1 = bbox(variant)
            aspects.add((x1 - x0) / (y1 - y0))
            positions.add((y0, x0))
        self.assertGreater(len(aspects), 5)
        self.assertGreater(len(positions), 5)

    def test_edge_touching_shape_keeps_its_contour_and_stays_in_frame(self):
        shape = np.zeros((86, 110), dtype=np.uint8)
        shape[:, :32] = 1
        shape[-20:, :] = 1
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[-86:, -110:] = shape

        for seed in range(12):
            variant = perturb_minority(mask, np.random.default_rng(seed))
            self.assert_binary_with_similar_area(variant, mask)
            self.assertFalse(is_rectangle(variant))
            self.assertNotEqual(cropped_shape(variant), cropped_shape(mask))
            self.assertFalse(variant[:300, :].any())
            self.assertFalse(variant[:, :300].any())
            y0, y1, x0, x1 = bbox(variant)
            self.assertTrue(variant[y0:y1, x0].all())
            self.assertTrue(variant[y1 - 1, x0:x1].all())
            self.assertEqual(variant[y0, x1 - 1], 0)

    def test_one_pixel_survives_perturbation(self):
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[0, 0] = 1
        variant = perturb_minority(mask, np.random.default_rng(8))
        self.assert_binary_with_similar_area(variant, mask)
        self.assertEqual(cropped_shape(mask), cropped_shape(variant))
        self.assertNotEqual(bbox(mask), bbox(variant))

    def test_cropped_signature_distinguishes_dimensions_and_ignores_translation(self):
        shape = np.array([[1, 1, 0, 0], [1, 0, 0, 1]], dtype=np.uint8)
        mask128 = np.zeros((128, 128), dtype=np.uint8)
        mask128[10:12, 18:22] = shape
        mask512 = np.zeros((512, 512), dtype=np.uint8)
        mask512[-2:, -4:] = shape
        reshaped = shape.reshape(4, 2)

        self.assertEqual(cropped_shape(mask128), cropped_shape(mask512))
        self.assertEqual(cropped_shape(reshaped)[1], cropped_shape(mask128)[1])
        self.assertNotEqual(cropped_shape(reshaped), cropped_shape(mask128))
        self.assertEqual(cropped_shape(np.zeros_like(mask128)), ((0, 0), b""))

    def assert_binary_with_similar_area(self, variant, source):
        self.assertEqual(variant.shape, (512, 512))
        self.assertEqual(variant.dtype, np.uint8)
        self.assertLessEqual(int(variant.max()), 1)
        self.assertGreaterEqual(int(variant.sum()), 0.8 * source.sum())
        self.assertLessEqual(int(variant.sum()), 1.2 * source.sum())


if __name__ == "__main__":
    unittest.main()
