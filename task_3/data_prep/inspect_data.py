"""Look at the data (login GPU, ~1-2 min):
  1. ../output/inspect_montage.jpg - each row: 5 forget images of one class | 5 external photos the original model
     puts in that class with the highest confidence (open the jpg in VS Code). What are the 100 classes?
  2. confidence of the original model on forget images vs external photos
  3. near-duplicates between forget images and the external photos (same source dataset?)
  4. if ../output/external_labels.npy exists (ext_labels.py): do the model's 100 classes match mini-ImageNet's?

    python data_prep/inspect_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # shared modules (common, diagnostics, ...)

import json

import numpy as np
import torch
import torch.nn.functional as F

from common import OUTPUT_DIR, load_external_images, load_forget_images, load_model, predict


def thumbs(imgs, size):
    x = imgs.permute(0, 3, 1, 2).float()
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


def save_jpg(arr, path):
    try:
        from PIL import Image
        Image.fromarray(arr).save(path, quality=90)
        print(f"saved {path}")
    except ImportError:
        import torchvision
        p = str(path).replace(".jpg", ".png")
        torchvision.utils.save_image(torch.from_numpy(arr).permute(2, 0, 1).float() / 255, p)
        print(f"saved {p}")


def q(t):
    return " ".join(f"{v:.3f}" for v in torch.quantile(t.float(), torch.tensor([0.1, 0.5, 0.9])).tolist())


def main():
    forget = load_forget_images()
    ext = load_external_images()
    teacher = load_model()
    pf = F.softmax(predict(teacher, forget).float(), 1)
    pe = F.softmax(predict(teacher, ext).float(), 1)
    yf, ye = pf.argmax(1), pe.argmax(1)
    print(f"original max-prob quantiles 10/50/90%: forget {q(pf.max(1).values)} | external {q(pe.max(1).values)}")
    print(f"external photos with max-prob > 0.9: {(pe.max(1).values > 0.9).float().mean():.1%}; "
          f"classes receiving >= 0.5% of the photos: {int((torch.bincount(ye, minlength=100) >= 0.005 * len(ye)).sum())}"
          f"/100")

    # 1. montage
    size, rows = 96, []
    classes = torch.bincount(yf, minlength=100).argsort(descending=True)[:10]
    for c in classes.tolist():
        fi = (yf == c).nonzero().flatten()[:5]
        ei = pe[:, c].argsort(descending=True)[:5]
        fthumbs = list(thumbs(forget[fi], size)) if len(fi) else []
        fthumbs += [torch.zeros(3, size, size)] * (5 - len(fthumbs))  # pad rows with < 5 forget images
        parts = fthumbs + [torch.full((3, size, 8), 255.0)] + list(thumbs(ext[ei], size))
        rows.append(torch.cat(parts, 2))
        print(f"row {len(rows)}: class {c}  ({int((yf == c).sum())} forget images; best external photo prob "
              f"{pe[ei[0], c]:.2f})")
    grid = torch.cat(rows, 1).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    save_jpg(np.ascontiguousarray(grid), OUTPUT_DIR / "inspect_montage.jpg")

    # 3. near-duplicates (16x16 colour thumbnails, mean-centred cosine)
    def vec(imgs):
        v = thumbs(imgs, 16).flatten(1)
        v = v - v.mean(1, keepdim=True)
        return F.normalize(v, dim=1)
    vf = vec(forget).cuda() if torch.cuda.is_available() else vec(forget)
    best = []
    for i in range(0, len(ext), 2000):
        ve = vec(ext[i:i + 2000]).to(vf.device)
        best.append((vf @ ve.T).max(1).values)
    best = torch.stack(best, 1).max(1).values.cpu()
    print(f"forget -> nearest external photo, cosine quantiles 10/50/90%: {q(best)}; "
          f"near-duplicates (>0.98): {int((best > 0.98).sum())}/{len(best)}")

    # 4. class correspondence with mini-ImageNet labels
    path = OUTPUT_DIR / "external_labels.npy"
    if path.exists():
        lab = torch.from_numpy(np.load(path))
        assert len(lab) == len(ext), "label count does not match external_224.npz"
        cons, tops = [], []
        for L in lab.unique().tolist():
            m = lab == L
            cnt = torch.bincount(ye[m], minlength=100)
            tops.append(int(cnt.argmax()))
            cons.append(float(cnt.max() / m.sum()))
        cons = torch.tensor(cons)
        print(f"mini-ImageNet labels: {len(cons)} classes; share of each class going to the model's most common "
              f"class: median {cons.median():.2f}, 10/90% {q(cons)}; distinct model classes used as top: "
              f"{len(set(tops))}")
        names_p = path.with_suffix(".names.json")
        if names_p.exists():
            names = json.loads(names_p.read_text())
            order = cons.argsort(descending=True)
            ul = lab.unique().tolist()
            print("most consistent:", ", ".join(f"{names[ul[i]]}->{tops[i]} ({cons[i]:.2f})" for i in order[:8]))


if __name__ == "__main__":
    main()
