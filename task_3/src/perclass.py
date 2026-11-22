"""Per-class accuracy on the forget set (labels = original model's predictions). Spots class-level collapse:
a good unlearned model has every class somewhere in the middle (record: 0.15-0.87); a model that wipes whole
classes shows a block at 0 and a block at 1 (SSD with external-photo F_r: 10 classes at 0, 27 at ~1 -> 0.048).

    python src/perclass.py ../output/best/score0.1249_pm0.5_alpha1.75.pt ../output/ssd_*.pt
"""
import sys
from pathlib import Path

import torch

from common import load_forget_images, load_model, predict


def main():
    forget = load_forget_images()
    y = predict(load_model(), forget).argmax(1)
    for name in sys.argv[1:]:
        p = predict(load_model(Path(name)), forget).argmax(1)
        accs = torch.tensor([(p[y == c] == c).float().mean().item() for c in y.unique()])
        h = torch.histc(accs, bins=5, min=0, max=1).int().tolist()
        print(f"{Path(name).name}: per-class acc histogram [0-.2,.2-.4,.4-.6,.6-.8,.8-1] = {h}  "
              f"(classes <0.05: {int((accs < 0.05).sum())}, >0.95: {int((accs > 0.95).sum())})")


if __name__ == "__main__":
    main()
