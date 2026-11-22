"""What did the original model do with the generated pool? (login node, ~1 min)
For every class: forget images, generated images with prob >= 0.5, and the prompt names that land there most
(= a guess of the class name). Plus ../output/genpool_preview2.jpg: rows = forget classes (5 forget images | the 6
MOST CONFIDENT generated images), then untouched classes (11 most confident generated).

    python data_prep/genpool_stats.py > ~/runs/genpool_stats.txt; head -n 40 ~/runs/genpool_stats.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # shared modules (common, diagnostics, ...)

from collections import Counter

import numpy as np
import torch
from PIL import Image

from common import GENPOOL_FILE, OUTPUT_DIR, load_forget_images, load_model, predict
from gen_pool import STYLES, nouns


def main():
    d = np.load(GENPOOL_FILE)
    imgs, label, prob, pidx = d["images"], d["label"], d["prob"], d["prompt"]
    names = nouns()
    noun = pidx // len(STYLES)
    forget = load_forget_images()
    fl = predict(load_model(), forget).argmax(1).numpy()
    nf = np.bincount(fl, minlength=100)
    ok = prob >= 0.5
    print(f"{len(imgs)} generated, {ok.sum()} with prob >= 0.5")
    print("class | forget imgs | gen>=0.5 | top prompt names (count, mean prob)")
    order = np.argsort(-nf, kind="stable")
    for c in order:
        m = ok & (label == c)
        top = Counter(noun[m].tolist()).most_common(5)
        desc = ", ".join(f"{names[n]} ({k}, {prob[m & (noun == n)].mean():.2f})" for n, k in top)
        print(f"{c:3d} | {nf[c]:4d} | {m.sum():4d} | {desc}")
    # how spread are the names? (one name -> one class means the model sees the object, not just the style)
    purity = []
    for n in np.unique(noun[ok]):
        purity.append(np.bincount(label[ok & (noun == n)], minlength=100).max() / (ok & (noun == n)).sum())
    print(f"names whose confident images all go to the same class: {np.mean(np.array(purity) == 1):.0%}")

    size, cols, rows = 96, 12, []
    for c in [c for c in order if nf[c] > 0][:8]:
        g = np.nonzero(label == c)[0]
        g = g[np.argsort(-prob[g])][:6]
        rows.append([forget[i].numpy() for i in np.nonzero(fl == c)[0][:5]] + [None] + [imgs[i] for i in g])
    for c in [c for c in order if nf[c] == 0][:6]:
        g = np.nonzero(label == c)[0]
        rows.append([imgs[i] for i in g[np.argsort(-prob[g])][:11]])
    canvas = Image.new("RGB", (cols * size, len(rows) * size), "white")
    for r, row in enumerate(rows):
        for k, a in enumerate(row[:cols]):
            if a is not None:
                canvas.paste(Image.fromarray(np.asarray(a)).resize((size, size)), (k * size, r * size))
    canvas.save(OUTPUT_DIR / "genpool_preview2.jpg", quality=85)
    print(f"preview: {OUTPUT_DIR / 'genpool_preview2.jpg'}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
