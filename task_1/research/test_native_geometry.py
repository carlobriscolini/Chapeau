import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
import torch

from research.gpu_synth import CleanPool, SynthGenerator, blobs, to_grid128
from wmloc.utils import downsample_mask


class FakeEmbedder:
    nbits = 7

    def __init__(self):
        self.calls = []

    def embed(self, images, messages):
        self.calls.append((images.clone(), messages.clone()))
        level = 8 if images.shape[-2:] == (512, 512) else 17
        return images + level / 255


def write_coco(root, records, *, height=128, width=208):
    images, annotations = [], []
    rows, cols = np.indices((height, width))
    for image_id, segments in records.items():
        filename = f"cover-{image_id}.png"
        pixels = np.stack([cols % 128, rows % 128, np.full_like(rows, image_id % 128)], axis=-1)
        Image.fromarray(pixels.astype(np.uint8)).save(root / filename)
        images.append({"id": image_id, "file_name": filename})
        annotations.extend({"image_id": image_id, **segment} for segment in segments)
    path = root / "instances.json"
    path.write_text(json.dumps({"images": images, "annotations": annotations}))
    return path


def coverage(masks):
    return masks.reshape(len(masks), 128, 4, 128, 4).sum(dim=(2, 4))


class NativeGeometryTest(unittest.TestCase):
    triangle = [[20, 20, 119, 41, 81, 113]]

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_pool_keeps_ids_cover_pixels_and_legacy_object_order(self):
        rle_shape = np.zeros((128, 208), np.uint8, order="F")
        rle_shape[12:25, 80:101] = 1
        rle = coco_mask.encode(rle_shape)
        rle["counts"] = rle["counts"].decode("ascii")
        large = [[20, 0, 125, 0, 125, 120, 20, 120]]
        annotations = write_coco(self.root, {
            784: [{"segmentation": self.triangle, "iscrowd": 1}],
            42: [{"segmentation": rle}, {"segmentation": self.triangle}, {"segmentation": large}],
            190: [{"segmentation": self.triangle}],
        })
        seed = 11
        legacy = CleanPool(self.root, annotations, 2, seed=seed, device="cpu")
        native = CleanPool(self.root, annotations, 2, seed=seed, device="cpu", native_contours=True)
        ids = [42, 190, 784]
        np.random.default_rng(seed).shuffle(ids)
        self.assertEqual(native.image_ids, tuple(ids[:2]))
        self.assertEqual(native.image_ids, legacy.image_ids)
        torch.testing.assert_close(native.images, legacy.images)
        torch.testing.assert_close(native.n_objects, legacy.n_objects)
        self.assertEqual(native.objects.shape, (2, 8, 512, 64))
        self.assertEqual(native.objects.dtype, torch.uint8)

        row = native.image_ids.index(42)
        left = int(np.random.default_rng(seed + row).integers(0, 81))
        expected_rle = np.asarray(Image.fromarray(rle_shape[:, left:left + 128]).resize((512, 512), Image.Resampling.NEAREST))
        # The fixture contains one eligible >50% instance. Its rank and eligibility follow the legacy grid.
        self.assertEqual(int(native.n_objects[row]), 3)
        self.assertGreater(legacy.objects[row, 0].float().mean().item(), 0.5)
        unpacked = np.unpackbits(native.objects[row].numpy(), axis=-1, bitorder="little")
        self.assertGreater(unpacked[0].mean(), 0.5)
        np.testing.assert_array_equal(unpacked[2], expected_rle)
        self.assertFalse(unpacked[3:].any())

    def test_native_object_pixels_follow_cropped_polygon_and_official_grid(self):
        annotations = write_coco(self.root, {42: [{"segmentation": self.triangle}]})
        native = CleanPool(self.root, annotations, 1, seed=12, device="cpu", native_contours=True)
        legacy = CleanPool(self.root, annotations, 1, seed=12, device="cpu")
        left = int(np.random.default_rng(12).integers(0, 81))
        expected_polygon = [[(coordinate - (left if axis % 2 == 0 else 0)) * 4
                             for axis, coordinate in enumerate(self.triangle[0])]]
        expected = torch.from_numpy(coco_mask.decode(coco_mask.frPyObjects(expected_polygon, 512, 512))[:, :, 0]).bool()
        for pool in (legacy, native):
            with self.subTest(native=pool.native_contours):
                generator = SynthGenerator("trustmark", pool, FakeEmbedder(), seed=0, p_rect=0, p_object=1)
                images, target, full, messages = generator.sample(6, return_messages=True)
                clean = pool.images.float() / 255
                changed = (images != clean).any(dim=1)
                self.assertTrue(full.any() and (~full).any())
                self.assertEqual(target.shape, (6, 128, 128))
                self.assertEqual(target.dtype, torch.bool)
                np.testing.assert_array_equal(target.numpy(), np.stack([downsample_mask(m) for m in changed.numpy()]))
                torch.testing.assert_close(messages, generator.emb.calls[0][1])
                counts = coverage(changed)
                mixed = (counts > 0) & (counts < 16)
                self.assertEqual(bool(mixed.any()), pool.native_contours)
                if pool.native_contours:
                    torch.testing.assert_close(changed, expected ^ full[:, None, None])

    def test_native_union_combines_two_distinct_packed_contours(self):
        left = [[10, 20, 45, 20, 45, 80, 10, 80]]
        right = [[199, 123, 275, 123, 275, 224, 199, 224]]
        annotations = write_coco(self.root, {42: [{"segmentation": left}, {"segmentation": right}]},
                                 height=512, width=512)
        pool = CleanPool(self.root, annotations, 1, device="cpu", native_contours=True)
        generator = SynthGenerator("trustmark", pool, FakeEmbedder(), seed=0, p_rect=0, p_object=1)
        images, _, full = generator.sample(12)
        changed = (images != pool.images.float() / 255).any(dim=1)
        sampled = changed ^ full[:, None, None]

        left_mask = torch.zeros((512, 512), dtype=torch.bool)
        right_mask = torch.zeros_like(left_mask)
        left_mask[20:80, 10:45] = True
        right_mask[123:224, 199:275] = True
        matches = (sampled[:, None] == torch.stack((left_mask, right_mask, left_mask | right_mask))).flatten(2).all(dim=2)
        self.assertTrue(matches.any(dim=1).all())
        self.assertTrue(matches.any(dim=0).all())

    def test_blobs_threshold_native_field_before_the_grid_and_keep_rng_draws(self):
        frac = torch.tensor([0.05, 0.2, 0.45])
        generators = [torch.Generator().manual_seed(25), torch.Generator().manual_seed(25)]
        legacy = blobs(3, frac, generators[0], 7)
        native = blobs(3, frac, generators[1], 7, size=512)
        torch.testing.assert_close(generators[0].get_state(), generators[1].get_state())
        self.assertEqual(legacy.shape, (3, 128, 128))
        self.assertEqual(native.shape, (3, 512, 512))
        counts = coverage(native)
        self.assertTrue(((counts > 0) & (counts < 16)).flatten(1).any(dim=1).all())
        torch.testing.assert_close(native.float().mean(dim=(1, 2)), frac, atol=1 / 512**2, rtol=0)

    def test_full_embed_precedes_crop_embedding_with_same_returned_bits(self):
        annotations = write_coco(self.root, {42: [{"segmentation": self.triangle}]})
        samples = []
        for native in (False, True):
            pool = CleanPool(self.root, annotations, 1, seed=12, device="cpu", native_contours=native)
            embedder = FakeEmbedder()
            generator = SynthGenerator("trustmark", pool, embedder, seed=0, p_rect=1, p_object=0)
            images, targets, full, messages = generator.sample(6, return_messages=True)
            self.assertEqual(images.shape, (6, 3, 512, 512))
            self.assertEqual(messages.shape, (6, embedder.nbits))
            self.assertTrue(((messages == 0) | (messages == 1)).all())
            legacy_return = SynthGenerator("trustmark", pool, FakeEmbedder(), seed=0, p_rect=1, p_object=0).sample(6)
            self.assertEqual(len(legacy_return), 3)
            for old, new in zip(legacy_return, (images, targets, full)):
                torch.testing.assert_close(old, new)
            clean, full_bits = embedder.calls[0]
            self.assertEqual(clean.shape, images.shape)
            torch.testing.assert_close(full_bits, messages)
            residual = ((images - clean) * 255).round().to(torch.int64)
            np.testing.assert_array_equal(targets.numpy(), np.stack([downsample_mask(m) for m in (residual[:, 0] != 0).numpy()]))
            crop_indices = (~full).nonzero().flatten()
            self.assertGreater(len(crop_indices), 0)
            self.assertGreater(int(full.sum()), 0)
            self.assertEqual(len(embedder.calls), 1 + len(crop_indices))
            for index, (patch, bits) in zip(crop_indices, embedder.calls[1:]):
                torch.testing.assert_close(bits[0], messages[index])
                changed = residual[index, 0] != 0
                spatial = changed.nonzero()
                y0, x0 = spatial.min(dim=0).values.tolist()
                y1, x1 = (spatial.max(dim=0).values + 1).tolist()
                torch.testing.assert_close(patch[0], clean[index, :, y0:y1, x0:x1])
                self.assertEqual(int(changed.sum()), (y1 - y0) * (x1 - x0))
                self.assertTrue((residual[index, :, y0:y1, x0:x1] == 17).all())
            self.assertTrue(((residual[full] == 0) | (residual[full] == 8)).all())
            samples.append((images, targets, full, messages))

        legacy_images, legacy_targets, legacy_full, legacy_bits = samples[0]
        native_images, native_targets, native_full, native_bits = samples[1]
        torch.testing.assert_close(legacy_full, native_full)
        torch.testing.assert_close(legacy_bits, native_bits)
        torch.testing.assert_close(legacy_images[~legacy_full], native_images[~native_full])
        torch.testing.assert_close(legacy_targets[~legacy_full], native_targets[~native_full])

    def test_grid_includes_eight_of_sixteen_pixels(self):
        mask = torch.zeros((1, 512, 512), dtype=torch.bool)
        mask[0, :2, :8] = True
        mask[0, 0, 4] = False
        target = to_grid128(mask)
        self.assertTrue(target[0, 0, 0])
        self.assertFalse(target[0, 0, 1])
        self.assertEqual(int(target.sum()), 1)


if __name__ == "__main__":
    unittest.main()
