"""Minimal pure-PyTorch AutoencoderKL (the latent-diffusion VAE of SD 1/2, SDXL and FLUX.1) that loads diffusers-format
safetensors with strict key matching, so the probe needs no extra package (the venv's torch 2.5 is too old for the
current diffusers). Architecture: Rombach et al., CVPR 2022; parameter names follow diffusers' AutoencoderKL.

    ae = load_ae("stabilityai/sd-vae-ft-mse")            # or ("ostris/Flex.1-alpha", "vae") for the FLUX.1 AE
    rec = ae.reconstruct(x * 2 - 1)                      # x in [0, 1], B x 3 x H x W -> reconstruction in [-1, 1]"""
from __future__ import annotations

import json

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(32, c, eps=1e-6)


class Resnet(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.norm1, self.conv1 = _gn(cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.norm2, self.conv2 = _gn(cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.conv_shortcut = nn.Conv2d(cin, cout, 1) if cin != cout else None

    def forward(self, x):
        h = self.conv2(F.silu(self.norm2(self.conv1(F.silu(self.norm1(x))))))
        return (x if self.conv_shortcut is None else self.conv_shortcut(x)) + h


class Attn(nn.Module):
    """Single-head spatial self-attention of the VAE mid block (residual)."""

    def __init__(self, c: int):
        super().__init__()
        self.group_norm = _gn(c)
        self.to_q, self.to_k, self.to_v = nn.Linear(c, c), nn.Linear(c, c), nn.Linear(c, c)
        self.to_out = nn.ModuleList([nn.Linear(c, c)])

    def forward(self, x):
        b, c, h, w = x.shape
        t = self.group_norm(x).view(b, c, h * w).transpose(1, 2)
        o = F.scaled_dot_product_attention(self.to_q(t)[:, None], self.to_k(t)[:, None], self.to_v(t)[:, None])[:, 0]
        return x + self.to_out[0](o).transpose(1, 2).reshape(b, c, h, w)


class Mid(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.resnets = nn.ModuleList([Resnet(c, c), Resnet(c, c)])
        self.attentions = nn.ModuleList([Attn(c)])

    def forward(self, x):
        return self.resnets[1](self.attentions[0](self.resnets[0](x)))


class Down(nn.Module):
    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        self.conv = conv

    def forward(self, x):
        return self.conv(F.pad(x, (0, 1, 0, 1)))


class Up(nn.Module):
    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        self.conv = conv

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class Block(nn.Module):
    def __init__(self, cin: int, cout: int, n: int, resample: str | None):
        super().__init__()
        self.resnets = nn.ModuleList([Resnet(cin if i == 0 else cout, cout) for i in range(n)])
        if resample == "down":
            self.downsamplers = nn.ModuleList([Down(nn.Conv2d(cout, cout, 3, stride=2))])
        elif resample == "up":
            self.upsamplers = nn.ModuleList([Up(nn.Conv2d(cout, cout, 3, padding=1))])

    def forward(self, x):
        for r in self.resnets:
            x = r(x)
        for s in getattr(self, "downsamplers", []) or getattr(self, "upsamplers", []):
            x = s(x)
        return x


class Encoder(nn.Module):
    def __init__(self, ch: tuple, n: int, zc: int):
        super().__init__()
        self.conv_in = nn.Conv2d(3, ch[0], 3, padding=1)
        self.down_blocks = nn.ModuleList([Block(ch[max(i - 1, 0)], ch[i], n, "down" if i < len(ch) - 1 else None)
                                          for i in range(len(ch))])
        self.mid_block = Mid(ch[-1])
        self.conv_norm_out, self.conv_out = _gn(ch[-1]), nn.Conv2d(ch[-1], 2 * zc, 3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        for b in self.down_blocks:
            x = b(x)
        return self.conv_out(F.silu(self.conv_norm_out(self.mid_block(x))))


class Decoder(nn.Module):
    def __init__(self, ch: tuple, n: int, zc: int):
        super().__init__()
        rev = ch[::-1]
        self.conv_in = nn.Conv2d(zc, rev[0], 3, padding=1)
        self.mid_block = Mid(rev[0])
        self.up_blocks = nn.ModuleList([Block(rev[max(i - 1, 0)], rev[i], n + 1, "up" if i < len(ch) - 1 else None)
                                        for i in range(len(ch))])
        self.conv_norm_out, self.conv_out = _gn(ch[0]), nn.Conv2d(ch[0], 3, 3, padding=1)

    def forward(self, z):
        x = self.mid_block(self.conv_in(z))
        for b in self.up_blocks:
            x = b(x)
        return self.conv_out(F.silu(self.conv_norm_out(x)))


class AutoencoderKL(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        ch, n, self.zc = tuple(cfg["block_out_channels"]), cfg["layers_per_block"], cfg["latent_channels"]
        self.encoder, self.decoder = Encoder(ch, n, self.zc), Decoder(ch, n, self.zc)
        self.quant_conv = nn.Conv2d(2 * self.zc, 2 * self.zc, 1) if cfg.get("use_quant_conv", True) else None
        self.post_quant_conv = nn.Conv2d(self.zc, self.zc, 1) if cfg.get("use_post_quant_conv", True) else None

    def encode_mean(self, x):
        m = self.encoder(x)
        if self.quant_conv is not None:
            m = self.quant_conv(m)
        return m[:, :self.zc]

    def decode(self, z):
        return self.decoder(z if self.post_quant_conv is None else self.post_quant_conv(z))

    def reconstruct(self, x):
        return self.decode(self.encode_mean(x))


_OLD = {".query.": ".to_q.", ".key.": ".to_k.", ".value.": ".to_v.", ".proj_attn.": ".to_out.0."}


def load_ae(repo: str, subfolder: str | None = None) -> AutoencoderKL:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    cfg = json.loads(open(hf_hub_download(repo, "config.json", subfolder=subfolder)).read())
    sd = {}
    for k, v in load_file(hf_hub_download(repo, "diffusion_pytorch_model.safetensors", subfolder=subfolder)).items():
        for old, new in _OLD.items():
            k = k.replace(old, new)
        sd[k] = v.reshape(v.shape[0], -1) if (".to_" in k and k.endswith("weight") and v.ndim == 4) else v
    ae = AutoencoderKL(cfg)
    ae.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
    return ae.eval()
