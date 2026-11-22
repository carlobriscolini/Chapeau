"""Build a combined test submission and its manifest (commit, checkpoints, per-method source, fold-0 check).

Per method, either a trained segmenter checkpoint (research.train_seg) or the polarity fill (research.polarity_fill).
    python -m research.predict --name sub01 \
        --model stegastamp=outputs/seg_v1/stegastamp/best.pt --model trustmark=outputs/seg_v1/trustmark/best.pt
Methods without --model use the polarity fill. Writes outputs/submissions/<name>.pt and research/submissions/<name>.json
(the manifest is tracked in git, so every leaderboard entry maps to a commit and to checkpoint hashes)."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from research import folds as folds_mod
from research import submission
from research.decoders import Decoder
from research.metrics import miou_batch
from research.polarity_fill import bit_accuracy, fit_threshold
from research.train_seg import MESSAGE_ENCODER_PREFIX, build_model, predict as seg_predict, stack_messages
from research.zero_shot import METHODS
from wmloc.data import LocalisationDataset

MANIFESTS = Path(__file__).with_name("submissions")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def uint8_batch(items: list[dict], device: str = "cuda") -> torch.Tensor:
    return torch.from_numpy(np.stack([it["image"] for it in items])).permute(0, 3, 1, 2).contiguous().to(device)


def predict_items(net: torch.nn.Module, items: list[dict], *, bs: int = 32, device: str = "cuda") -> torch.Tensor:
    messages = stack_messages(items, device) if net.requires_messages else None
    return seg_predict(net, uint8_batch(items, device), bs=bs, messages=messages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--data", default="release")
    ap.add_argument("--model", action="append", default=[], help="method=checkpoint.pt[@encoder][+checkpoint2.pt[@encoder]...] (several: mean logits)")
    ap.add_argument("--encoder", default="resnet34")
    a = ap.parse_args()
    models = dict(m.split("=", 1) for m in a.model)
    fold_of = folds_mod.load()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "research/predict.py",
                                 "research/train_seg.py", "research/wm_encoder.py", "research/decoders.py",
                                 "research/polarity_fill.py", "wmloc"], capture_output=True, text=True).stdout.strip())
    manifest = {"name": a.name, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "commit": commit,
                "code_dirty": dirty, "methods": {}}
    names, masks = [], []
    for method in METHODS:
        tr = LocalisationDataset(a.data, "train", methods=[method]); tr = [tr[i] for i in range(len(tr))]
        te = LocalisationDataset(a.data, "test", methods=[method]); te = [te[i] for i in range(len(te))]
        val = [it for it in tr if fold_of[it["image_name"]] == 0]
        gt = np.stack([it["mask128"] for it in val])
        if method in models:
            val_logits, test_logits, members = 0, 0, []
            parts = models[method].split("+")                  # several checkpoints: average their logits
            for part in parts:
                spec, _, encoder = part.partition("@"); encoder = encoder or a.encoder
                if encoder.startswith(MESSAGE_ENCODER_PREFIX) and encoder != f"{MESSAGE_ENCODER_PREFIX}{method}":
                    ap.error(f"{encoder} does not match the watermark method {method}")
                ckpt = Path(spec)
                net = build_model(encoder, pretrained=False).cuda().to(memory_format=torch.channels_last)
                net.load_state_dict(torch.load(ckpt, map_location="cuda", weights_only=True)); net.eval()
                val_logits = val_logits + predict_items(net, val).cpu().numpy() / len(parts)
                test_logits = test_logits + predict_items(net, te).cpu().numpy() / len(parts)
                member = {"encoder": encoder, "checkpoint": str(ckpt), "sha256": sha256(ckpt)}
                log = ckpt.with_name("log.json")
                if log.exists():
                    member["train_args"] = json.loads(log.read_text())["args"]
                members.append(member)
                del net
            val_pred, test_pred = val_logits > 0, test_logits > 0
            info = {"source": "segmenter", **members[0]} if len(members) == 1 else \
                {"source": "segmenter_ensemble", "members": members}
        else:
            dec = Decoder(method)
            acc_tr, acc_te = bit_accuracy(dec, tr, method), bit_accuracy(dec, te, method)
            full = np.array([bool(it["full_watermark"]) for it in tr])
            f = np.array([fold_of[it["image_name"]] for it in tr])
            t_val = fit_threshold(acc_tr[f != 0], full[f != 0]); thr = fit_threshold(acc_tr, full)
            val_pred = np.broadcast_to((acc_tr[f == 0] > t_val)[:, None, None], gt.shape)
            test_pred = np.broadcast_to((acc_te > thr)[:, None, None], (len(te), 128, 128))
            info = {"source": "polarity_fill", "bit_accuracy_threshold": thr}
            del dec
        torch.cuda.empty_cache()
        info["fold0_miou"] = float(miou_batch(val_pred, gt).mean()); info["fold0_images"] = len(val)
        info["test_watermarked_fraction"] = float(np.mean(test_pred))
        manifest["methods"][method] = info
        names += [it["image_name"] for it in te]; masks.append(np.asarray(test_pred, np.uint8))
        print(method, json.dumps({k: v for k, v in info.items() if k != "train_args"}), flush=True)
    out = Path("outputs/submissions") / f"{a.name}.pt"
    manifest["file"] = submission.write(names, np.concatenate(masks), out, a.data)
    manifest["file"]["sha256"] = sha256(out)
    manifest["fold0_miou_mean"] = float(np.mean([m["fold0_miou"] for m in manifest["methods"].values()]))
    MANIFESTS.mkdir(exist_ok=True)
    (MANIFESTS / f"{a.name}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("commit", "code_dirty", "fold0_miou_mean", "file")}, indent=2))


if __name__ == "__main__":
    main()
