import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

from research.predict import predict_items
from research.train_seg import build_model, extra_split, mix_training_sources, predict, real_split
from research.wm_encoder import WatermarkSegNet


class FixtureDecoder(nn.Module):
    def __init__(self, method, device):
        super().__init__()
        self.nbits = 2
        self.net = nn.BatchNorm2d(3).to(device).eval().requires_grad_(False)

    def per_position_logits(self, x):
        return self.net(x)[:, :2]


def fixture_stages(x):
    return [F.adaptive_avg_pool2d(x[:, :1], size).expand(-1, channels, -1, -1)
            for size, channels in zip((8, 4, 2, 1), (256, 512, 1024, 2048), strict=True)]


def write_training_data(root, rows, *, extra=False):
    train = root / "train"
    (train / "images").mkdir(parents=True)
    if not extra:
        (train / "masks").mkdir()
    metadata, masks = [], []
    for name, pixel, message, fold in rows:
        image = np.full((4, 4, 3), pixel, dtype=np.uint8)
        Image.fromarray(image).save(train / "images" / f"{name}.png")
        mask = np.full((4, 4), int(message[0]), dtype=np.uint8)
        masks.append(mask)
        if not extra:
            Image.fromarray(mask * 255).save(train / "masks" / f"{name}.png")
        row = {"image_name": name, "method": "trustmark", "message": message, "full_watermark": False}
        if extra:
            row["source_fold"] = fold
        metadata.append(row)
    (train / "metadata.json").write_text(json.dumps(metadata))
    np.savez(train / "masks_128.npz", names=[row[0] for row in rows], masks=masks)


class MessageConditioningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.addClassCleanup(torch.set_num_threads, previous)

    def test_spatial_agreement_reverses_with_the_message_and_reaches_the_mask(self):
        with patch("research.wm_encoder.Decoder", FixtureDecoder), \
                patch.object(WatermarkSegNet, "_stages", staticmethod(fixture_stages)):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(3)
                control = build_model("wm:trustmark", device="cpu").eval()
                control_rng = torch.random.get_rng_state()
                torch.manual_seed(3)
                model = build_model("wmmsg:trustmark", device="cpu").eval()
                self.assertTrue(torch.equal(torch.random.get_rng_state(), control_rng))
            image = torch.linspace(0, 1, 64).reshape(1, 1, 8, 8).repeat(1, 3, 1, 1)
            bits = torch.ones(1, 2, dtype=torch.uint8)
            with torch.no_grad():
                torch.testing.assert_close(model(image, bits), control(image), rtol=0, atol=0)
            model.train()
            self.assertFalse(model.message_decoder.training)
            self.assertTrue(model.body.training)
            agreement = model.local_agreement(image, bits)
            self.assertGreater(float(agreement.std()), 0)
            torch.testing.assert_close(model.local_agreement(image, 1 - bits), -agreement, rtol=0, atol=0)
            expected = model(image, bits)
            self.assertEqual(expected.shape, (1, 128, 128))
            expected.mean().backward()
            self.assertGreater(model.message_projection.weight.grad.abs().sum().item(), 0)
            self.assertTrue(all(parameter.grad is None for parameter in model.message_decoder.parameters()))
            torch.optim.SGD(model.message_projection.parameters(), lr=.1).step()
            self.assertFalse(torch.equal(model(image, bits), model(image, 1 - bits)))
            with self.assertRaisesRegex(ValueError, "require the known watermark messages"):
                model(image)

    def test_legacy_rgb_checkpoint_predicts_bitwise_identically(self):
        image = torch.arange(3 * 64 * 64).reshape(1, 3, 64, 64).byte()
        baseline = build_model("resnet18", pretrained=False).eval()
        x = (image.float() / 255).contiguous(memory_format=torch.channels_last)
        checkpoint = io.BytesIO()
        torch.save(baseline.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored = build_model("resnet18", pretrained=False)
        restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
        item = {"image": image[0].permute(1, 2, 0).numpy()}
        with torch.no_grad():
            expected = baseline(x)
        torch.testing.assert_close(predict(restored, image, 1), expected, rtol=0, atol=0)
        torch.testing.assert_close(predict_items(restored, [item], device="cpu"), expected, rtol=0, atol=0)

    def test_training_sources_keep_bits_with_the_original_images_and_folds(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            release = [("real_train", 64, "10", 1), ("real_val", 96, "01", 0)]
            extras = [("extra_train", 128, "11", 1), ("extra_val", 160, "00", 0)]
            write_training_data(root / "release", release)
            write_training_data(root / "extra", extras, extra=True)
            folds = {row[0]: row[-1] for row in release}
            with patch("research.train_seg.folds_mod.load", return_value=folds):
                real, val = real_split(str(root / "release"), "trustmark", 0, "cpu", return_messages=True)
                self.assertTrue(all(len(split) == 3 for split in real_split(str(root / "release"), "trustmark", 0, "cpu")))
            extra = extra_split(str(root / "extra"), "trustmark", 0, "cpu", return_messages=True)
            self.assertEqual(len(extra_split(str(root / "extra"), "trustmark", 0, "cpu")), 2)

        self.assertEqual(val[0][0, 0, 0, 0].item(), 96)
        self.assertEqual(val[3].tolist(), [[0, 1]])
        online = (torch.full((1, 3, 4, 4), .125), torch.zeros((1, 4, 4), dtype=torch.bool), torch.tensor([[0, 0]]))
        image, mask, bits = mix_training_sources(online, (real[0], real[1], real[3]), extra, lambda x: x + .125)
        torch.testing.assert_close(image[:, 0, 0, 0], torch.tensor([.25, 64 / 255, 128 / 255 + .125]))
        self.assertEqual(mask[:, 0, 0].tolist(), [False, True, True])
        self.assertEqual(bits.tolist(), [[0, 0], [1, 0], [1, 1]])

    def test_inference_slices_actual_images_and_messages_together(self):
        class PairModel(nn.Module):
            requires_messages = True

            def forward(self, x, messages):
                return x[:, 0, 0, 0] * 255 + messages[:, 0] + 2 * messages[:, 1]

        items = [{"image": np.full((4, 4, 3), pixel, dtype=np.uint8), "message": np.array(bits, dtype=np.uint8)}
                 for pixel, bits in ((10, [0, 1]), (20, [1, 0]), (30, [1, 1]))]
        logits = predict_items(PairModel(), items, bs=2, device="cpu")
        torch.testing.assert_close(logits, torch.tensor([12., 21., 33.]), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
