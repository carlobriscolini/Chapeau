from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from research.coco_fresh_pair import image_index_records, paired_templates
from research.protocol_v3 import FOLDS, InvertedMessages, comparisons, training_settings
from research.trustmark_holdout import read_json


class ProtocolTest(unittest.TestCase):
    def test_cover_derangement_keeps_each_recipe_and_cover_once(self):
        rows = [{"source_coco_id": i, "image_name": f"recipe{i}"} for i in range(8)]
        pairs = paired_templates(rows, len(rows), np.random.default_rng(11))
        self.assertEqual(len(pairs), len(rows))
        for side in zip(*pairs):
            self.assertCountEqual(side, rows)
        self.assertTrue(all(left["source_coco_id"] != right["source_coco_id"] for left, right in pairs))

    def test_read_image_index_independently_of_geometries(self):
        stream = BytesIO(b'{"info":{"description":"COCO"}, "images" : [{"id":11}], "annotations":[')
        self.assertEqual(image_index_records(stream), [{"id": 11}])

    def test_arms_vary_one_argument(self):
        root = Path("paired")
        base = training_settings(root, "pixelseal", FOLDS[0], "reused")
        expected = {"native": "native_contours", "fresh": "extra_dir", "message": "encoder"}
        for arm, key in expected.items():
            settings = training_settings(root, "pixelseal", FOLDS[0], arm)
            self.assertEqual([name for name in settings if settings[name] != base[name]], [key])

    def test_message_inversion_preserves_pixels_and_source_bits(self):
        image = np.array([23], dtype=np.uint8)
        bits = np.array([1, 0, 1, 1], dtype=np.uint8)
        original = {"image": image, "message": bits}
        flipped = InvertedMessages([original])[0]
        self.assertIs(flipped["image"], image)
        np.testing.assert_array_equal(flipped["message"], [0, 1, 0, 0])
        np.testing.assert_array_equal(original["message"], [1, 0, 1, 1])

    def test_pooled_gain_cannot_pass_with_only_one_fold(self):
        def evaluation(a, b):
            return {"scores": {"miou": (a + b) / 2}, "per_image_miou": {"a": a, "b": b}}

        method = "pixelseal"
        first, second = FOLDS
        with TemporaryDirectory() as path:
            out = Path(path)
            results = {(method, first, "reused"): evaluation(0.4, 0.4),
                       (method, first, "native"): evaluation(0.6, 0.4)}
            comparisons(out, results)
            report = read_json(out / "report.json")["comparisons"]["pixelseal:native"]
            self.assertFalse(report["positive_on_both_folds"])
            results[method, second, "reused"] = evaluation(0.4, 0.4)
            results[method, second, "native"] = evaluation(0.8, 0.4)
            comparisons(out, results)
            report = read_json(out / "report.json")["comparisons"]["pixelseal:native"]
            self.assertTrue(report["positive_on_both_folds"])
            self.assertEqual(report["pooled_n"], 4)
            self.assertAlmostEqual(report["pooled_delta_miou"], 0.15)


if __name__ == "__main__":
    unittest.main()
