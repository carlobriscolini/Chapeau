"""Shared helpers: model / data loading and server-identical preprocessing + prediction.

Data files live one directory above this repo (<task dir>/{model.pt,forget_images.npz}).
Override with the env var DATA_DIR if needed.
"""
import os
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F

REPO_DIR = Path(__file__).resolve().parents[1]  # this folder (src/ is one level down)
DATA_DIR = Path(os.environ.get("DATA_DIR", REPO_DIR.parent))
MODEL_FILE = DATA_DIR / "model.pt"
FORGET_IMAGES_FILE = DATA_DIR / "forget_images.npz"
EXTERNAL_IMAGES_FILE = DATA_DIR / "output" / "external_224.npz"  # built by prepare_external.py
SIBLINGS_FILE = DATA_DIR / "output" / "siblings.npz"  # built by gen_siblings.py (variations of forget images)
EXTGEN_FILE = DATA_DIR / "output" / "extgen.npz"  # built by gen_siblings.py (photos re-rendered by the generator)
INVERTED_FILE = DATA_DIR / "output" / "inverted.npz"  # built by invert.py (model-inversion anchors)
INVERTED_HOLDOUT = 400  # last N inverted images, never anchors
GENPOOL_FILE = DATA_DIR / "output" / "genpool.npz"  # built by gen_pool.py (cinematic-style generated images)
GENPOOL_T_FILE = DATA_DIR / "output" / "genpool_t.npz"  # gen_pool.py --targeted (round 2, never held out)
GENPOOL_P_FILE = DATA_DIR / "output" / "genpool_p.npz"  # gen_pool.py --targeted --photo (round 3, photographic look)
GENPOOL_HOLDOUT = 1500  # last N (after class-balanced selection + shuffle), never anchors
# held-out parts, never used as retain anchors, so the drift checks are honest
EXTERNAL_HOLDOUT = 2000  # last N external photos
EXTGEN_HOLDOUT = 1000  # last N extgen images
SIB_HOLDOUT_SRC = 3000  # siblings of forget images with index >= this
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", DATA_DIR / "output"))

NUM_CLASSES = 100
IMAGE_SIZE = 224
TEMPERATURE = 2.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# bf16 on A100 (compute nodes); fp16 on GPUs without bf16 support (e.g. Quadro RTX 8000 on login nodes)
AMP_DTYPE = torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16


def build_model() -> torch.nn.Module:
    """ViT-B/16 at 224x224 with LayerScale (init 1e-4), exactly as used by the server."""
    return timm.create_model("vit_base_patch16_224", pretrained=False,
                             num_classes=NUM_CLASSES, init_values=1e-4)


def load_model(path: Path = MODEL_FILE) -> torch.nn.Module:
    model = build_model()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.to(DEVICE).eval()


def load_forget_images(path: Path = FORGET_IMAGES_FILE) -> torch.Tensor:
    """uint8 [N, 224, 224, 3], raw pixels, channels last."""
    with np.load(path, allow_pickle=False) as data:
        images = torch.from_numpy(data["images"].copy())
    assert images.dtype == torch.uint8 and tuple(images.shape[1:]) == (IMAGE_SIZE, IMAGE_SIZE, 3), images.shape
    return images


def load_external_images(path: Path = EXTERNAL_IMAGES_FILE):
    """External unlabeled natural images uint8 [M,224,224,3], or None if prepare_external.py was not run."""
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        return torch.from_numpy(data["images"].copy())


def holdout_size(n: int, n_holdout: int) -> int:
    """Number of images kept out of training (never anchors) for drift checks: n_holdout, at most 20% of n."""
    return min(n_holdout, n // 5)


def load_siblings(path: Path = SIBLINGS_FILE):
    """(images uint8 [M,224,224,3], src long [M] = index of the forget image) or (None, None)."""
    if not path.exists():
        return None, None
    with np.load(path, allow_pickle=False) as data:
        return torch.from_numpy(data["images"].copy()), torch.from_numpy(data["src"].copy()).long()


def load_inverted(path: Path = INVERTED_FILE):
    """Model-inversion images uint8 [M,224,224,3] (classes absent from the forget set), or None."""
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        return torch.from_numpy(data["images"].copy())


def load_extgen(path: Path = EXTGEN_FILE):
    """Generator-style re-renderings of external photos uint8 [M,224,224,3], or None."""
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        return torch.from_numpy(data["images"].copy())


def load_genpool(min_prob: float = 0.5, cap: int = 200, path: Path = GENPOOL_FILE):
    """Generated cinematic-style images that the original model classifies with prob >= min_prob, at most `cap`
    per class (the most confident), shuffled with a fixed seed. -> (images uint8 [M,224,224,3], labels) or (None, None)."""
    if not path.exists():
        return None, None
    with np.load(path, allow_pickle=False) as data:
        prob = torch.from_numpy(data["prob"].copy()).float()
        lab = torch.from_numpy(data["label"].copy()).long()
        keep = []
        for c in lab.unique().tolist():
            idx = ((lab == c) & (prob >= min_prob)).nonzero().flatten()
            keep.append(idx[prob[idx].argsort(descending=True)][:cap])
        keep = torch.cat(keep)
        keep = keep[torch.randperm(len(keep), generator=torch.Generator().manual_seed(0))]
        images = torch.from_numpy(data["images"][keep.numpy()])
    return images, lab[keep]


def preprocess(images_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 [B,224,224,3] -> float32 [B,3,224,224] in [-1,1]. Identical to the server."""
    x = images_uint8.to(DEVICE).permute(0, 3, 1, 2).float() / 255.0
    return (x - 0.5) / 0.5


@torch.no_grad()
def predict(model: torch.nn.Module, images_uint8: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    """Logits [N,100] computed like the server (eval mode, fp16 autocast on GPU)."""
    model.eval()
    out = []
    for i in range(0, len(images_uint8), batch_size):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            out.append(model(preprocess(images_uint8[i:i + batch_size])).float().cpu())
    return torch.cat(out)


def soft_probs(logits: torch.Tensor) -> torch.Tensor:
    """Class distribution used by the metric: softmax(logits / 2)."""
    return F.softmax(logits / TEMPERATURE, dim=1)


def kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Per-sample KL(p || q) for probability rows."""
    return (p * (p.clamp_min(1e-12).log() - q.clamp_min(1e-12).log())).sum(1)


def save_submission(model: torch.nn.Module, path: Path) -> None:
    """Plain float32 state dict that loads strict=True into build_model(); checks NaN/Inf and size."""
    sd = {k: v.detach().cpu().float() for k, v in model.state_dict().items()}
    for k, v in sd.items():
        assert torch.isfinite(v).all(), f"non-finite values in {k}"
    build_model().load_state_dict(sd, strict=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, path)
    size_mb = path.stat().st_size / 1e6
    assert size_mb < 450, f"{path} is {size_mb:.0f} MB (> 450)"
    print(f"Saved {path} ({size_mb:.0f} MB)")
