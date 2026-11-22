"""Differentiable, message-conditioned views of the three official extractors.

Every decoder maps images B x 3 x H x W in [0, 1] (any H, W; the official wrappers' resizing is reproduced) to
B x nbits logits with autograd enabled, so the evidence for the *known* message can be traced back to pixels.
StegaStamp is only released as ONNX: its decoder subgraph is converted to PyTorch with onnx2torch."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from wmloc.watermarks import load_watermark

STEGA_LOGITS = "gen_decoder/decoder_out/BiasAdd:0"


def _stegastamp_torch(models_dir: Path) -> torch.nn.Module:
    import onnx
    from onnx2torch import convert
    graph = onnx.load(str(models_dir / "stegastamp" / "stega_stamp_logits.onnx"))
    return convert(onnx.utils.Extractor(graph).extract_model(["image"], [STEGA_LOGITS]))


class Decoder(torch.nn.Module):
    """logits(x) for one method, plus native per-position bit logits where the architecture exposes them."""

    def __init__(self, method: str, models_dir: str | Path = "models", device: str = "cuda"):
        super().__init__()
        self.method, self.device = method, device
        models_dir = Path(models_dir)
        if method == "stegastamp":
            self.wm = None
            self.net = _stegastamp_torch(models_dir).to(device).eval()
            self.nbits = 100
        else:
            self.wm = load_watermark(method, models_dir, device)
            self.nbits = self.wm.nbits
            self.net = self.wm.det if method == "pixelseal" else self.wm.net
        for p in self.net.parameters():
            p.requires_grad_(False)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        if self.method == "pixelseal":
            return self.net(self.wm._resize(x))[:, 1:].float()
        if self.method == "trustmark":
            return self.net(self.wm._prep(x)).float()
        x = F.interpolate(x.float(), size=(400, 400), mode="bicubic", align_corners=False, antialias=True)
        return self.net(x.clamp(0, 1).permute(0, 2, 3, 1)).float()

    @torch.no_grad()
    def per_position_logits(self, x: torch.Tensor, size: int | None = None) -> torch.Tensor | None:
        """B x nbits x h x w logits before the global pooling, with the input resized to `size` (None: the official
        working resolution). StegaStamp has a dense head after flattening, so it has no such map."""
        x = x.to(self.device)
        if self.method == "pixelseal":
            x = F.interpolate(x, size=(size or 256,) * 2, mode="bilinear", align_corners=False, antialias=True)
            det = self.net
            emb = det.pixel_decoder.output_upscaling(det.convnext(det.preprocess(x)))
            lin = det.pixel_decoder.linear
            return (torch.einsum("bchw,oc->bohw", emb, lin.weight) + lin.bias[None, :, None, None])[:, 1:].float()
        if self.method == "trustmark":
            n = self.net
            x = F.interpolate(x, size=(size or self.wm.resolution,) * 2, mode="bilinear", align_corners=False,
                              antialias=True) * 2 - 1
            x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
            x = n.layer4(n.layer3(n.layer2(n.layer1(x))))
            return (torch.einsum("bchw,oc->bohw", x, n.fc.weight) + n.fc.bias[None, :, None, None]).float()
        return None


def signed(messages: torch.Tensor) -> torch.Tensor:
    """bits {0, 1} -> {-1, +1}, the sign a logit must have to agree with the known message."""
    return messages.float() * 2 - 1
