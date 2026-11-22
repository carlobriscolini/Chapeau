"""Is model.pt a fine-tune of a public timm ViT-B/16 checkpoint? (login node, ~2 min, downloads ~350 MB per candidate)

For every candidate: relative distance ||W_model - W_base|| / ||W_model|| per block (head excluded, different classes).
Values << 1 (e.g. 0.05-0.3) = fine-tuned from that checkpoint; ~1.4 = unrelated weights.

    HF_HOME=../output/hf_tmp python experiments/check_base.py      # then: rm -rf ../output/hf_tmp
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # shared modules (common, diagnostics, ...)

import sys

import timm
import torch

from common import MODEL_FILE

CANDIDATES = sys.argv[1:] or [
    "deit3_base_patch16_224.fb_in1k",  # DeiT-III uses LayerScale (init 1e-4 in timm), like model.pt
    "deit3_base_patch16_224.fb_in22k_ft_in1k",
    "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "vit_base_patch16_224.augreg_in21k",
    "vit_base_patch16_224.augreg_in1k",
    "vit_base_patch16_224.orig_in21k",
]


def main():
    sd = torch.load(MODEL_FILE, map_location="cpu", weights_only=True)
    gam = {k: float(v.abs().mean()) for k, v in sd.items() if k.endswith(".gamma")}
    if gam:  # LayerScale init is 1e-4: values still ~1e-4 => blocks barely used; large => trained long with LS
        print("LayerScale |gamma| mean per tensor (first 6):",
              " ".join(f"{k.split('.')[1]}.{k.split('.')[2]}:{v:.2e}" for k, v in list(gam.items())[:6]))
    for name in CANDIDATES:
        try:
            base = timm.create_model(name, pretrained=True).state_dict()
        except Exception as exc:  # noqa: BLE001
            print(f"{name}: could not load ({exc})")
            continue
        rows = []
        for k, w in sd.items():
            if k.startswith("head") or k not in base or base[k].shape != w.shape:
                continue
            rows.append((k, float((w - base[k]).norm() / w.norm().clamp_min(1e-12))))
        if not rows:
            print(f"{name}: no matching tensors")
            continue
        blocks = {}
        for k, d in rows:
            b = k.split(".")[1] if k.startswith("blocks.") else k.split(".")[0]
            blocks.setdefault(b, []).append(d)
        summary = "  ".join(f"{b}:{sum(v) / len(v):.3f}" for b, v in blocks.items())
        print(f"\n{name}: mean rel. distance {sum(d for _, d in rows) / len(rows):.3f}\n  {summary}")
        missing = [k for k in sd if not k.startswith("head") and k not in base]
        if missing:
            print(f"  keys not in base: {missing[:5]}")


if __name__ == "__main__":
    main()
