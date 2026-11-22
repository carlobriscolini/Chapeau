"""Labels of ../output/external_224.npz -> ../output/external_labels.npy (same files, order and permutation as
prepare_external.py). Needs internet; run in a throw-away env like prepare_external.py:

    HF_HOME=../output/hf_tmp uv run --no-project --with pyarrow --with numpy --with huggingface_hub \
        python data_prep/ext_labels.py && rm -rf ../output/hf_tmp
"""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "timm/mini-imagenet"
FILES = ["data/test-00000-of-00002.parquet", "data/test-00001-of-00002.parquet",
         "data/validation-00000-of-00003.parquet", "data/validation-00001-of-00003.parquet"]
N = 10000  # prepare_external.py --n (default)
OUT = Path(__file__).resolve().parents[2] / "output" / "external_labels.npy"


def main():
    labels, names = [], None
    for f in FILES:
        path = hf_hub_download(REPO, f, repo_type="dataset")
        table = pq.read_table(path)
        if names is None:  # class names from the HF features metadata, if present
            try:
                meta = json.loads(table.schema.metadata[b"huggingface"])
                names = meta["info"]["features"]["label"]["names"]
            except Exception:  # noqa: BLE001
                names = None
        labels += table.column("label").to_pylist()
        if len(labels) >= N:
            break
    lab = np.array(labels[:N], dtype=np.int64)
    lab = lab[np.random.default_rng(0).permutation(len(lab))]
    np.save(OUT, lab)
    print(f"saved {OUT}: {len(lab)} labels, {len(np.unique(lab))} distinct classes")
    if names:
        OUT.with_suffix(".names.json").write_text(json.dumps(names))
        print("class names saved (first 10):", names[:10])


if __name__ == "__main__":
    main()
