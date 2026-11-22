import unittest

import numpy as np
import torch

from research.metrics import miou_batch
from research.oracle_bench import best_threshold, calibrate_maps, prior_table, rank_minorities
from research.zero_shot import maps_for_batch


class OracleBenchTest(unittest.TestCase):
    def test_threshold_oracle_matches_brute_force_with_ties_and_empty_classes(self):
        score = np.array([[.2, .2, -.5], [.7, .1, -.5]])
        masks = [np.array([[1, 0, 0], [1, 0, 1]], bool), np.zeros(score.shape, bool), np.ones(score.shape, bool)]
        thresholds = [float("-inf"), *np.unique(score)]
        for mask in masks:
            brute = max(miou_batch((score > value)[None], mask[None])[0] for value in thresholds)
            self.assertAlmostEqual(best_threshold(score, mask), brute)

    def test_fast_rank_scores_match_actual_masks_in_both_polarities(self):
        rng = np.random.default_rng(97)
        scores = rng.integers(-2, 3, (5, 8, 8)).astype(float)
        target = rng.random(scores.shape) > .8
        table = prior_table({"signal": scores}, target)
        for (_, fraction, full), expected in table.items():
            actual = np.stack([rank_minorities(score, full, fraction) for score in scores])
            np.testing.assert_allclose(miou_batch(actual, target), expected)

    def test_fold_zero_predictions_do_not_use_fold_zero_labels(self):
        rng = np.random.default_rng(1)
        scores = rng.normal(size=(12, 8, 8))
        masks = scores > .3
        full = np.tile([False, True], 6)
        fractions = np.where(full, .99, .5)
        folds = np.repeat([0, 1, 2], 4)
        pred, background, settings = calibrate_maps({"map": scores}, masks, full, fractions, folds)
        masks[folds == 0] = ~masks[folds == 0]
        full[folds == 0] = ~full[folds == 0]
        changed, changed_background, changed_settings = calibrate_maps({"map": scores}, masks, full, fractions, folds)
        np.testing.assert_array_equal(pred[:4], changed[:4])
        np.testing.assert_array_equal(background[:4], changed_background[:4])
        self.assertEqual(settings[0], changed_settings[0])

    def test_requesting_one_map_preserves_its_values_and_global_evidence(self):
        class FakeDecoder:
            @staticmethod
            def logits(x):
                return x[:, :2].mean(dim=(2, 3)) - .5

            @staticmethod
            def per_position_logits(x, size):
                return x[:, :2] - .5

        torch.set_num_threads(2)
        images = torch.rand(1, 3, 40, 40, generator=torch.Generator().manual_seed(7))
        signed_bits = torch.tensor([[-1., 1.]])
        maps, evidence = maps_for_batch(FakeDecoder(), images, signed_bits)
        selected, selected_evidence = maps_for_batch(FakeDecoder(), images, signed_bits, requested={"native_work"})
        self.assertEqual(set(selected), {"native_work"})
        np.testing.assert_array_equal(selected["native_work"], maps["native_work"])
        for name in evidence:
            np.testing.assert_array_equal(evidence[name], selected_evidence[name])
