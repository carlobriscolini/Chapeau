import unittest

import torch

from research.post_watermark import PostWatermarkAugment, _TRANSFORMS
from research.train_seg import mix_training_sources


class PostWatermarkAugmentTest(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(17)
        self.images = torch.rand(8, 3, 32, 32, generator=generator)

    def test_disabled_augmentation_has_an_exact_identity_path(self):
        augment = PostWatermarkAugment(seed=3, probability=0)
        self.assertIs(augment(self.images), self.images)

    def test_seeded_batches_match_and_the_input_is_untouched(self):
        original = self.images.clone()
        aug_a = PostWatermarkAugment(seed=21, probability=0.5)
        aug_b = PostWatermarkAugment(seed=21, probability=0.5)
        for _ in range(3):
            result = aug_a(self.images)
            torch.testing.assert_close(result, aug_b(self.images), rtol=0, atol=0)
            self.assertEqual(result.shape, self.images.shape)
            self.assertEqual(result.dtype, self.images.dtype)
            self.assertEqual(result.device, self.images.device)
            self.assertTrue(((0 <= result) & (result <= 1)).all())
        torch.testing.assert_close(self.images, original, rtol=0, atol=0)

    def test_a_local_pattern_stays_at_the_same_coordinates(self):
        source = torch.zeros(3, 32, 32)
        source[:, 19:24, 4:9] = 0.8
        augment = PostWatermarkAugment(seed=7)
        for operation in _TRANSFORMS:
            with self.subTest(operation=operation):
                output = augment._apply(source, operation)
                torch.testing.assert_close(output[0] > 0.4, source[0] > 0.4)

    def test_noise_matches_the_calibrated_strength(self):
        image = torch.full((3, 64, 64), 0.5)
        changed = PostWatermarkAugment(seed=7)._apply(image, "noise")
        self.assertAlmostEqual((changed - image).std().item(), 0.5 / 255, delta=0.0001)

    def test_training_mixes_both_synthetic_sources_and_preserves_real_images(self):
        online = (self.images[0:1], torch.tensor([[[True, False]]]))
        real = ((self.images[1:2] * 255).round().byte(), torch.tensor([[[False, True]]]))
        extra = ((self.images[2:3] * 255).round().byte(), torch.tensor([[[True, True]]]))
        expected = torch.cat([online[0], real[0].float() / 255, extra[0].float() / 255])
        masks = torch.cat([online[1], real[1], extra[1]])

        control = PostWatermarkAugment(seed=7, probability=0)
        unmodified, control_masks = mix_training_sources(online, real, extra, control)
        torch.testing.assert_close(unmodified, expected, rtol=0, atol=0)
        torch.testing.assert_close(control_masks, masks)

        active = PostWatermarkAugment(seed=7, probability=1)
        modified, active_masks = mix_training_sources(online, real, extra, active)
        self.assertFalse(torch.equal(modified[0], expected[0]))
        torch.testing.assert_close(modified[1], expected[1], rtol=0, atol=0)
        self.assertFalse(torch.equal(modified[2], expected[2]))
        torch.testing.assert_close(active_masks, masks)
