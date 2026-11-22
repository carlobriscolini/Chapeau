"""Submit one or more state dicts to the hackathon server (same request as the organisers' submission.py).

Run on the LOGIN node:
    export UNLEARN_SERVER_URL="http://..."   # evaluation server (given by the organisers)
    export UNLEARN_API_KEY="..."             # never commit the key
    python src/submit.py ../output/sub_a.pt ../output/sub_b.pt ...
Files are sent one after the other; when the server answers "submitted too recently, wait N seconds" the script
sleeps N seconds and retries. Every score is appended to ../output/scores.tsv (time, file, score).
Tip: run it inside `tmux` or with `nohup ... &` so it survives closing the terminal.
"""
import os
import re
import sys
import time
from pathlib import Path

import requests

TASK_ID = "31-unlearning"
SERVER_URL = os.environ.get("UNLEARN_SERVER_URL", "").rstrip("/")  # organisers' evaluation server
SCORES_FILE = Path(__file__).resolve().parents[2] / "output" / "scores.tsv"


def submit_once(api_key: str, file: Path):
    """Returns (status, payload): status in {"ok", "wait", "error"}."""
    with file.open("rb") as f:
        try:
            reply = requests.post(f"{SERVER_URL}/submit/{TASK_ID}", headers={"X-API-Key": api_key},
                                  files={"file": (file.name, f)}, timeout=(10, 600))
        except requests.RequestException as exc:
            return "error", f"Could not reach the server: {exc}"
    try:
        body = reply.json()
    except ValueError:
        return "error", f"HTTP {reply.status_code}: {reply.text[:300]}"
    if reply.status_code == 429:
        detail = str(body.get("detail", body))
        m = re.search(r"(\d+)\s*seconds", detail)
        return "wait", int(m.group(1)) if m else 600
    if reply.status_code != 200:
        return "error", f"Rejected (HTTP {reply.status_code}): {body.get('detail', body)}"
    if body.get("status") == "success":
        return "ok", body
    return "error", f"{body.get('status')}: {body.get('message')}"


def main() -> int:
    api_key = os.environ.get("UNLEARN_API_KEY", "").strip()
    if not SERVER_URL:
        print("Set UNLEARN_SERVER_URL first (the organisers' evaluation server, e.g. http://host)", file=sys.stderr)
        return 1
    if not api_key:
        print("Set UNLEARN_API_KEY first: export UNLEARN_API_KEY=...", file=sys.stderr)
        return 1
    files = [Path(a) for a in sys.argv[1:]]
    if not files:
        print("Usage: python src/submit.py <file.pt> [more.pt ...]", file=sys.stderr)
        return 1
    for file in files:
        if not file.is_file() or file.suffix.lower() not in (".pt", ".pth"):
            print(f"Not a .pt/.pth file: {file}", file=sys.stderr)
            return 1

    failed = 0
    for file in files:
        while True:
            print(f"[{time.strftime('%H:%M:%S')}] Submitting {file.name} ...", flush=True)
            status, payload = submit_once(api_key, file)
            if status == "wait":
                print(f"  server says wait {payload}s, sleeping", flush=True)
                time.sleep(payload + 5)
                continue
            break
        if status == "ok":
            score = payload.get("score")
            print(f"  Score (public split): {score}   previous best: {payload.get('previous_best')}  "
                  f"improved: {payload.get('improved')}", flush=True)
            extra = {k: v for k, v in payload.items() if k not in ("score", "previous_best", "improved", "status")}
            if extra:  # anything else the server returns (e.g. forget / retain components)
                print(f"  full reply: {extra}", flush=True)
            SCORES_FILE.parent.mkdir(parents=True, exist_ok=True)
            with SCORES_FILE.open("a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{file.name}\t{score}\n")
        else:
            print(f"  {payload}", file=sys.stderr, flush=True)
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
