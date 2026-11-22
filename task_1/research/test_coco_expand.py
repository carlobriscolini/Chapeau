import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from research.coco_expand import choose_mask, cropped_shape, make_example, minority_mask, quotas
from wmloc.utils import downsample_mask


class MaskExpansionTest(unittest.TestCase):
    def test_quotas_preserve_the_total_and_round_the_three_size_bands(self):
        sizes = {"below_1pct": 13, "1_to_5pct": 39, "at_least_5pct": 48}
        self.assertEqual(quotas(sizes, 17), {"below_1pct": 2, "1_to_5pct": 7, "at_least_5pct": 8})

    def test_a_new_minority_shape_survives_the_grid_in_both_polarities(self):
        original = np.zeros((512, 512), dtype=np.uint8)
        original[20:52, 30:62] = 1
        variant = np.zeros_like(original)
        variant[40:64, 40:80] = 1
        cover = Image.new("RGB", (512, 512))

        class FakeWatermark:
            nbits = 100

            def embed(self, image, bits):
                return image + 1

        for full in (False, True):
            with self.subTest(full=full):
                parent = {"source_coco_id": 123, "source_fold": 4, "source_train_recipe": "rule",
                          "image_name": "cover", "message": "0" * 100,
                          "full_watermark": full, "method": "trustmark"}
                parent_mask = minority_mask(original, full)
                parent_grid = downsample_mask(parent_mask)
                args = (parent, parent_mask, parent_grid, (0, 0, 512, 512), {}, np.random.default_rng(7), set())
                with patch("research.coco_expand.perturb_minority", return_value=variant):
                    selected = choose_mask(*args, {False: set(), True: set()})
                    np.testing.assert_array_equal(selected.mask, minority_mask(variant, full))
                    used = {False: set(), True: set()}
                    used[full].add(cropped_shape(minority_mask(selected.grid, full)))
                    self.assertIsNone(choose_mask(*args, used))
                image, row = make_example(FakeWatermark(), parent, selected, cover, np.random.default_rng(7))
                np.testing.assert_array_equal(np.asarray(image)[..., 0], selected.mask)
                self.assertEqual(row["source_fold"], 4)
                self.assertEqual(row["parent_image_name"], parent["image_name"])
                self.assertNotEqual(row["message"], parent["message"])
