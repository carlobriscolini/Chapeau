"""Batched GPU re-implementation of the three official embedders (wmloc.watermarks.*.embed work image by image on
CPU/PIL). Inputs and outputs are float tensors B x 3 x H x W in [0, 1] on the GPU, quantised to 8 bits like the
official uint8 outputs. `python -m research.gpu_embed` checks parity with the official embedders."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from wmloc.watermarks import load_watermark


def _stegastamp_encoder(models_dir: Path) -> torch.nn.Module:
    import onnx
    from onnx2torch import convert
    graph = onnx.load(str(models_dir / "stegastamp" / "stega_stamp_logits.onnx"))
    return convert(onnx.utils.Extractor(graph).extract_model(["secret", "image"], ["residual"]))


class GpuEmbedder:
    def __init__(self, method: str, models_dir: str | Path = "models", device: str = "cuda"):
        self.method, self.device = method, device
        models_dir = Path(models_dir)
        if method == "stegastamp":
            self.enc = _stegastamp_encoder(models_dir).to(device).eval()
            self.nbits = 100
        else:
            self.wm = load_watermark(method, models_dir, device)
            self.nbits = self.wm.nbits

    @torch.no_grad()
    def embed(self, x: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device).float(); bits = bits.to(self.device).float()
        h, w = x.shape[-2:]
        if self.method == "pixelseal":
            out = self.wm.model.embed(x, msgs=bits, is_video=False)["imgs_w"].clamp(0, 1)
            return (out * 255).round() / 255
        if self.method == "trustmark":
            tm = self.wm.tm
            res = tm.model_resolution_enc
            cover = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False, antialias=True) * 2 - 1
            stego, _ = tm.encoder(cover, bits)
            r = stego.clamp(-1, 1) - cover
            r = F.interpolate(r - r.mean(dim=(2, 3), keepdim=True), size=(h, w), mode="bilinear")
            out = (r + x * 2 - 1).clamp(-1, 1) * 127.5 + 127.5
            return out.floor() / 255
        small = F.interpolate(x, size=(400, 400), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        r = self.enc(bits, small.permute(0, 2, 3, 1)) * 255                # B x 400 x 400 x 3
        r = (r - r.mean(dim=(1, 2), keepdim=True)).permute(0, 3, 1, 2)
        r = F.interpolate(r, size=(h, w), mode="bilinear", align_corners=False)
        return (x * 255 + r).clamp(0, 255).floor() / 255


def parity(n: int = 8) -> None:
    from research.decoders import Decoder, signed
    from wmloc.data import LocalisationDataset
    ds = LocalisationDataset("release", "test")
    imgs = [ds[i]["image"] for i in range(n)]
    for method in ("pixelseal", "trustmark", "stegastamp"):
        g = GpuEmbedder(method); ref = load_watermark(method, device="cuda"); dec = Decoder(method)
        bits = torch.randint(0, 2, (n, g.nbits))
        x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float().cuda() / 255
        out = g.embed(x, bits)
        official = torch.from_numpy(np.stack([ref.embed(im, b.tolist()) for im, b in zip(imgs, bits)])).permute(0, 3, 1, 2).float().cuda() / 255
        diff = ((out - official).abs() * 255)
        with torch.no_grad():
            acc = ((dec.logits(out) > 0) == (signed(bits).cuda() > 0)).float().mean().item()
        print(f"{method}: |gpu - official| mean {diff.mean():.3f} max {diff.max():.0f} (8-bit levels), "
              f"share exact {(diff < 0.5).float().mean():.3f}, gpu bit acc {acc:.3f}", flush=True)


if __name__ == "__main__":
    parity()
