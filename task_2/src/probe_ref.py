"""Send a probe prompt file to the reference endpoints ONLY (ref-*). Resume-safe.

    python probe_ref.py probes_v1.jsonl --repeats 4 [--out data/ref_v1.jsonl] [--endpoints ref-01,ref-02]
"""
import argparse
from pathlib import Path

import common
import query_reference as qr

ap = argparse.ArgumentParser()
ap.add_argument("prompts")
ap.add_argument("--repeats", type=int, default=4)
ap.add_argument("--out", default=None)
ap.add_argument("--endpoints", default=None)
ap.add_argument("--max-tokens", type=int, default=256)
a = ap.parse_args()

prompts = qr.read_prompts(Path(a.prompts))
out = Path(a.out) if a.out else common.DATA / f"ref_{Path(a.prompts).stem}.jsonl"
if a.endpoints:
    eps = a.endpoints.split(",")
else:
    eps = sorted(e for e in common.budgets() if e.startswith("ref-"))
assert eps and all(e.startswith("ref-") for e in eps), eps
print("endpoints:", eps, "->", out)
common.query_safe(prompts, out, eps, repeats=a.repeats, max_tokens=a.max_tokens)
