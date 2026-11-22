"""Step 1: how much localisation signal do the official extractors give for free, given the known message?

For every train image, computes soft 128 x 128 presence maps (higher = more likely watermarked):
  native_work  per-position agreement with the message, before global pooling, at the official working resolution
  native_512   same, with the 512 x 512 image fed without downscaling (finer map)          [PixelSeal, TrustMark]
  occlusion    drop of the agreement when a 32 x 32 block is replaced by its blurred version (16 x 16 blocks)
  hp_energy    local high-pass energy of the image alone (model-free reference)
and global polarity evidence (bit accuracy and mean signed logit of the whole image) against `full_watermark`.

    python -m research.zero_shot --data release --out outputs/zero_shot
Writes maps.npz (float16 maps + names), metrics.json and report.md. All scores are in-sample train diagnostics."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from research.decoders import Decoder, signed
from research.metrics import auc_per_image, best_global_threshold, miou_batch, oracle_fraction_miou
from wmloc.data import LocalisationDataset

METHODS = ("pixelseal", "trustmark", "stegastamp")
BATCH = {"pixelseal": 32, "trustmark": 32, "stegastamp": 16}
OCC_GRID = 16


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    r = max(1, int(3 * sigma)); t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2)); k = k / k.sum()
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def to_grid(m: torch.Tensor) -> torch.Tensor:
    """B x h x w -> B x 128 x 128 (bilinear up, or 4 x 4 average down from 512)."""
    m = m[:, None].float()
    m = F.avg_pool2d(m, m.shape[-1] // 128) if m.shape[-1] > 128 else F.interpolate(m, size=(128, 128), mode="bilinear", align_corners=False)
    return m[:, 0]


def agreement(dec: Decoder, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return (dec.logits(x) * s).mean(1)


def maps_for_batch(dec: Decoder, x: torch.Tensor, s: torch.Tensor,
                   requested: set[str] | None = None) -> tuple[dict, dict]:
    """Return global evidence and optionally only the selected spatial maps."""
    out, glob = {}, {}
    with torch.no_grad():
        lg = dec.logits(x)
        glob["bit_acc"] = ((lg > 0).float() == (s > 0).float()).float().mean(1)
        glob["agreement"] = (lg * s).mean(1)
        for name, size in (("native_work", None), ("native_512", 512)):
            if requested is not None and name not in requested:
                continue
            pp = dec.per_position_logits(x, size)
            if pp is not None:
                out[name] = to_grid((pp * s[:, :, None, None]).mean(1))
        if requested is None or "hp_energy" in requested:
            hp = x - gaussian_blur(x, 2.0)
            out["hp_energy"] = to_grid(gaussian_blur((hp ** 2).sum(1, keepdim=True), 4.0)[:, 0])

        if requested is None or "occlusion" in requested:
            blurred, base, c = gaussian_blur(x, 3.0), glob["agreement"], x.shape[-1] // OCC_GRID
            occ = torch.zeros(len(x), OCC_GRID, OCC_GRID, device=x.device)
            for i in range(OCC_GRID):
                for j in range(OCC_GRID):
                    xo = x.clone(); sl = (slice(None), slice(None), slice(i * c, (i + 1) * c), slice(j * c, (j + 1) * c))
                    xo[sl] = blurred[sl]
                    occ[:, i, j] = base - agreement(dec, xo, s)
            out["occlusion"] = to_grid(occ)
    return {k: v.float().cpu().numpy() for k, v in out.items()}, {k: v.float().cpu().numpy() for k, v in glob.items()}


def polarity_stats(score: np.ndarray, full: np.ndarray) -> dict:
    auc = auc_per_image(score[None], full[None].astype(np.uint8))[0]
    ts = np.unique(score)
    acc = max(((score > t) == full).mean() for t in np.concatenate([ts, [ts.min() - 1]]))
    return {"auc_full_watermark": float(auc), "best_threshold_accuracy": float(acc)}


def evaluate(method: str, items: list[dict], dec: Decoder | None = None) -> tuple[dict, dict, dict]:
    """Metrics, maps and global evidence for items carrying image, message, mask128 and full_watermark."""
    t0, n = time.time(), len(items)
    dec = dec or Decoder(method)
    maps, glob = {}, {}
    for b in range(0, n, BATCH[method]):
        chunk = items[b:b + BATCH[method]]
        x = torch.from_numpy(np.stack([it["image"] for it in chunk])).permute(0, 3, 1, 2).float().cuda() / 255
        s = signed(torch.from_numpy(np.stack([it["message"] for it in chunk]))).cuda()
        m, g = maps_for_batch(dec, x, s)
        for k, v in m.items(): maps.setdefault(k, []).append(v)
        for k, v in g.items(): glob.setdefault(k, []).append(v)
        print(f"::progress {b + len(chunk)}/{n} {method}", flush=True)
    maps = {k: np.concatenate(v) for k, v in maps.items()}; glob = {k: np.concatenate(v) for k, v in glob.items()}
    gt = np.stack([it["mask128"] for it in items]); full = np.array([bool(it["full_watermark"]) for it in items])
    res = {"images": n, "seconds": round(time.time() - t0, 1),
           "polarity": {k: polarity_stats(v, full) for k, v in glob.items()},
           "bit_acc_mean": {"full_watermark_true": float(glob["bit_acc"][full].mean()),
                            "full_watermark_false": float(glob["bit_acc"][~full].mean())},
           "constant": {"all_clean": float(miou_batch(np.zeros_like(gt), gt).mean()),
                        "oracle_polarity_fill": float(miou_batch(np.broadcast_to(full[:, None, None], gt.shape), gt).mean())},
           "maps": {}}
    for k, v in maps.items():
        auc = auc_per_image(v, gt)
        thr, miou = best_global_threshold(v, gt)
        res["maps"][k] = {"auc_mean": float(np.nanmean(auc)),
                          "auc_mean_bg_clean": float(np.nanmean(auc[~full])),
                          "auc_mean_bg_marked": float(np.nanmean(auc[full])),
                          "miou_best_global_z_threshold": miou, "z_threshold": thr,
                          "miou_oracle_fraction": oracle_fraction_miou(v, gt)}
    return res, maps, glob


def run(data: str, out: Path, methods=METHODS, limit: int | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    report, saved = {}, {}
    for method in methods:
        ds = LocalisationDataset(data, "train", methods=[method])
        items = [ds[i] for i in range(min(len(ds), limit or len(ds)))]
        res, maps, glob = evaluate(method, items)
        for k, v in maps.items(): saved[f"{method}/{k}"] = v.astype(np.float16)
        for k, v in glob.items(): saved[f"{method}/global_{k}"] = v
        saved[f"{method}/names"] = np.array([it["image_name"] for it in items])
        report[method] = res
        torch.cuda.empty_cache()
    np.savez_compressed(out / "maps.npz", **saved)
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    (out / "report.md").write_text(markdown(report))
    return report


def markdown(report: dict) -> str:
    lines = ["# Zero-shot presence maps (train, in-sample diagnostics)", ""]
    for method, r in report.items():
        lines += [f"## {method} ({r['images']} images, {r['seconds']} s)", "",
                  f"- all clean mIoU {r['constant']['all_clean']:.3f}; fill with true polarity {r['constant']['oracle_polarity_fill']:.3f}",
                  f"- whole-image bit accuracy: {r['bit_acc_mean']['full_watermark_true']:.3f} (marked background) vs "
                  f"{r['bit_acc_mean']['full_watermark_false']:.3f} (clean background)"]
        for k, p in r["polarity"].items():
            lines.append(f"- polarity from {k}: AUC {p['auc_full_watermark']:.3f}, best accuracy {p['best_threshold_accuracy']:.3f}")
        lines += ["", "| map | AUC | AUC clean bg | AUC marked bg | mIoU best z-thr | mIoU oracle fraction |", "|---|---:|---:|---:|---:|---:|"]
        for k, m in sorted(r["maps"].items(), key=lambda kv: -kv[1]["auc_mean"]):
            lines.append(f"| {k} | {m['auc_mean']:.3f} | {m['auc_mean_bg_clean']:.3f} | {m['auc_mean_bg_marked']:.3f} | "
                         f"{m['miou_best_global_z_threshold']:.3f} | {m['miou_oracle_fraction']:.3f} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="release"); ap.add_argument("--out", default="outputs/zero_shot")
    ap.add_argument("--methods", nargs="*", default=list(METHODS)); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    print(markdown(run(a.data, Path(a.out), a.methods, a.limit)))
