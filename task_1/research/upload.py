"""Upload a built submission to the leaderboard and record the answer in its manifest.

Refuses a file that fails research.submission.validate or whose sha256 differs from the manifest, and a second upload
of a manifest that already has a leaderboard entry. Credentials come from .env (WMLOC_BASE_URL, WMLOC_API_KEY) and
are never printed.

    python -m research.upload research/submissions/sub02_jureca.json"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import requests

from research import submission

TASK_ID = "34-watermark-localisation-image"


def env(path: Path = Path(".env")) -> dict:
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line and not line.startswith("#"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest"); ap.add_argument("--data", default="release")
    a = ap.parse_args()
    mpath = Path(a.manifest); manifest = json.loads(mpath.read_text())
    if "leaderboard" in manifest:
        raise SystemExit(f"already uploaded: {manifest['leaderboard']}")
    sub = Path(manifest["file"]["path"])
    submission.validate(sub, a.data)
    digest = hashlib.sha256(sub.read_bytes()).hexdigest()
    if digest != manifest["file"]["sha256"]:
        raise SystemExit("the submission file changed since its manifest was written")
    cfg = env()
    with sub.open("rb") as f:
        resp = requests.post(f"{cfg['WMLOC_BASE_URL'].rstrip('/')}/submit/{TASK_ID}",
                             headers={"X-API-Key": cfg["WMLOC_API_KEY"]},
                             files={"file": (sub.name, f, "application/octet-stream")}, timeout=(10, 300))
    try:
        body = resp.json()
    except ValueError:
        body = {"raw_text": resp.text[:500]}
    print(resp.status_code, json.dumps(body))
    resp.raise_for_status()
    manifest["leaderboard"] = {"submission_id": body.get("submission_id"), "public_score": body.get("score"), "response": body, "task": TASK_ID,
                               "submitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
