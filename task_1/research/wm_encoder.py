"""Segmenters whose encoder is the official watermark extractor itself (weights already tuned to the watermark).

  wm:pixelseal   PixelSeal ConvNeXt-V2 detector, image resized to its 256 working resolution
  wm:trustmark   TrustMark ResNet-50 decoder, image resized to its 245 working resolution
Four stages (strides 4..32 of the working resolution) feed an FPN; the stride-4 logits are resized to the official
128 x 128 grid. The extractor is fine-tuned with the rest of the network. The wmmsg: modes also fuse a spatial
agreement map from a frozen official decoder and the known message."""
from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from research.decoders import Decoder, signed


class WatermarkSegNet(nn.Module):
    def __init__(self, method: str, width: int = 128, *, message_conditioned: bool = False, device: str = "cuda"):
        super().__init__()
        self.method = method
        self.requires_messages = message_conditioned
        if method not in ("pixelseal", "trustmark"):
            raise ValueError(f"no differentiable extractor encoder for {method}")
        decoder = Decoder(method, device=device)
        net = deepcopy(decoder.net) if message_conditioned else decoder.net
        if method == "pixelseal":
            self.res, self.chans = 256, [96, 192, 384, 768]
            self.pre, self.body = net.preprocess, net.convnext
        else:
            self.res, self.chans = 245, [256, 512, 1024, 2048]
            self.body = net
        for p in self.body.parameters():
            p.requires_grad_(True)
        self.lat = nn.ModuleList(nn.Conv2d(c, width, 1) for c in self.chans)
        self.head = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
                                  nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
                                  nn.Conv2d(width, 1, 1))
        if message_conditioned:
            self.message_decoder = decoder.eval()
            # Start from the RGB logits and keep the training RNG aligned with the control arm.
            with torch.random.fork_rng(devices=[]):
                self.message_projection = nn.Conv2d(1, width, 1, bias=False, device="cpu")
            nn.init.zeros_(self.message_projection.weight)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.requires_messages:
            self.message_decoder.eval()
        return self

    def local_agreement(self, x: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        """Mean signed decoder logit at each native spatial position."""
        if messages.shape != (len(x), self.message_decoder.nbits):
            raise ValueError(f"expected one {self.message_decoder.nbits}-bit {self.method} message per image")
        logits = self.message_decoder.per_position_logits(x)
        signs = signed(messages).to(logits.device)
        return (logits * signs[:, :, None, None]).mean(dim=1, keepdim=True)

    def _stages(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = F.interpolate(x, size=(self.res, self.res), mode="bilinear", align_corners=False, antialias=True)
        feats = []
        if self.method == "pixelseal":
            x = self.pre(x)
            for down, stage in zip(self.body.downsample_layers, self.body.stages):
                x = stage(down(x)); feats.append(x)
        else:
            n = self.body; x = n.maxpool(n.relu(n.bn1(n.conv1(x * 2 - 1))))
            for k in ("layer1", "layer2", "layer3", "layer4"):
                x = getattr(n, k)(x); feats.append(x)
        return feats

    def forward(self, x: torch.Tensor, messages: torch.Tensor | None = None) -> torch.Tensor:
        if self.requires_messages and messages is None:
            raise ValueError("wmmsg encoders require the known watermark messages")
        feats = self._stages(x)
        p = self.lat[3](feats[3])
        for i in (2, 1, 0):
            p = self.lat[i](feats[i]) + F.interpolate(p, size=feats[i].shape[-2:], mode="bilinear", align_corners=False)
        if self.requires_messages:
            agreement = self.local_agreement(x, messages)
            agreement = F.interpolate(agreement, size=p.shape[-2:], mode="bilinear", align_corners=False)
            p = p + self.message_projection(agreement)
        return F.interpolate(self.head(p), size=(128, 128), mode="bilinear", align_corners=False)[:, 0]


class HighResWatermarkSegNet(WatermarkSegNet):
    """WatermarkSegNet plus a full-resolution branch for fine boundaries and small regions.

    The extractor still sees its native working resolution (its features end on a 64 x 64 grid); a light ImageNet
    ResNet-34 stem on the 512 input adds stride-4 (128 x 128) and stride-8 features, so the head runs on the official
    128 grid instead of upsampling 64 x 64 logits."""

    def __init__(self, method: str, width: int = 128, *, pretrained: bool = True, device: str = "cuda"):
        super().__init__(method, width, device=device)
        import timm
        self.fine = timm.create_model("resnet34", pretrained=pretrained, features_only=True, out_indices=(1, 2))
        self.fine_lat = nn.ModuleList(nn.Conv2d(c, width, 1) for c in self.fine.feature_info.channels())
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor, messages: torch.Tensor | None = None) -> torch.Tensor:
        feats = self._stages(x)
        p = self.lat[3](feats[3])
        for i in (2, 1, 0):
            p = self.lat[i](feats[i]) + F.interpolate(p, size=feats[i].shape[-2:], mode="bilinear", align_corners=False)
        s4, s8 = self.fine((x - self.mean) / self.std)                 # 128 x 128 and 64 x 64 for a 512 input
        q = self.fine_lat[1](s8) + F.interpolate(p, size=s8.shape[-2:], mode="bilinear", align_corners=False)
        q = self.fine_lat[0](s4) + F.interpolate(q, size=s4.shape[-2:], mode="bilinear", align_corners=False)
        return F.interpolate(self.head(q), size=(128, 128), mode="bilinear", align_corners=False)[:, 0]
