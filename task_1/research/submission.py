"""Write and validate a submission file against the rules of the brief (no upload here)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

MAX_BYTES = 20 * 1000 * 1000      # the brief says 20 MB, the starter README 25 MB: keep the stricter one


def write(names: list[str], masks: np.ndarray, path: Path, data: str = "release") -> dict:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    masks = np.asarray(masks)
    assert masks.shape == (len(names), 128, 128), masks.shape
    assert set(np.unique(masks).tolist()) <= {0, 1}
    torch.save({"image_names": list(names), "masks": torch.from_numpy(masks.astype(bool))}, path)
    return validate(path, data)


def validate(path: Path, data: str = "release") -> dict:
    sub = torch.load(path, weights_only=True)
    expected = {m["image_name"] for m in json.loads((Path(data) / "test" / "metadata.json").read_text())}
    names, masks = sub["image_names"], sub["masks"]
    assert set(sub) == {"image_names", "masks"}
    assert len(names) == len(set(names)) == len(expected) and set(names) == expected, "names must match the test set"
    assert tuple(masks.shape) == (len(expected), 128, 128) and masks.dtype in (torch.bool, torch.uint8)
    size = Path(path).stat().st_size
    assert size <= MAX_BYTES, f"{size} bytes > {MAX_BYTES}"
    return {"path": str(path), "images": len(names), "bytes": size,
            "watermarked_fraction": float(masks.float().mean())}
