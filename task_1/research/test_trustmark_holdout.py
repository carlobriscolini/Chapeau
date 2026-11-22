import unittest

import numpy as np

from research.trustmark_holdout import fit_area_rule


class TrustMarkHoldoutTest(unittest.TestCase):
    def test_held_out_masks_and_messages_cannot_change_calibration(self):
        backgrounds = np.array([False, True] * 6)
        masks = np.broadcast_to(backgrounds[:, None, None], (12, 16, 16)).copy()
        for i in range(12):
            masks[i, :1 + i % 4, :5] = not backgrounds[i]
        bits = np.where(backgrounds, .99, .52)
        calibration = np.arange(len(masks)) >= 4
        expected = fit_area_rule(masks, backgrounds, bits, calibration)

        backgrounds[~calibration] = ~backgrounds[~calibration]
        masks[~calibration] = True
        bits[~calibration] = .76
        self.assertEqual(fit_area_rule(masks, backgrounds, bits, calibration), expected)
