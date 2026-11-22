from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from research.coco_augment import object_mask, synthesise
from research.overlap import ImageIndex


class CoverOverlapTest(unittest.TestCase):
    def test_finds_identical_and_reencoded_images(self):
        rng = np.random.default_rng(412)
        original = Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)).resize((160, 160))
        different = Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)).resize((160, 160))
        index = ImageIndex([(("test", "known"), original)])
        self.assertTrue(index.matches(original.copy())[0]["exact_rgb"])
        with BytesIO() as buffer:
            original.resize((90, 90)).save(buffer, format="JPEG", quality=84)
            buffer.seek(0)
            reencoded = index.matches(Image.open(buffer))
        self.assertEqual(len(reencoded), 1)
        self.assertFalse(reencoded[0]["exact_rgb"])
        self.assertEqual(index.matches(different), [])


class CocoMaskTest(unittest.TestCase):
    def test_polygon_is_transformed_to_the_selected_crop(self):
        polygon = {"id": 4, "area": 1250, "segmentation": [[60, 40, 110, 40, 60, 90]]}
        result = object_mask([polygon], (50, 25, 150, 125), 32_000, np.random.default_rng(1))
        self.assertIsNotNone(result)
        mask, annotation_id = result
        self.assertEqual(annotation_id, 4)
        self.assertEqual(mask[100, 75], 1)
        self.assertEqual(mask[0, 0], 0)
        self.assertEqual(mask[300, 300], 0)

    def test_both_backgrounds_preserve_pixels_outside_the_mask(self):
        class FakeWatermark:
            nbits = 100

            def embed(self, image, _):
                return image + np.uint8(1)

        with TemporaryDirectory() as directory:
            photo = Path(directory) / "000001.jpg"
            Image.new("RGB", (512, 512)).save(photo)
            patch = np.zeros((512, 512), dtype=np.uint8)
            patch[80:120, 60:130] = 1
            for full, mask in ((False, patch), (True, 1 - patch)):
                with self.subTest(full=full):
                    donor = {"image_name": "donor", "full_watermark": full}
                    marked, label, row = synthesise(FakeWatermark(), "trustmark", photo, donor, mask, {}, 0)
                    expected_rect = (0, 512, 0, 512) if full else (80, 120, 60, 130)
                    np.testing.assert_array_equal(np.asarray(marked)[..., 0], mask)
                    np.testing.assert_array_equal(label, mask)
                    self.assertEqual(row["embedding_rect"], expected_rect)
                    self.assertEqual(len(row["message"]), 100)
