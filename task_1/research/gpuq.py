"""Local job queue shared by every agent (Codex, Claude) working on this machine.

One job holds the local GPU at a time, and a job declares the data it reads or writes so that no job reads a
dataset while another one rewrites it. Locks are kernel flocks: they are released when the process dies, so a crashed
job never blocks the queue.

    python -m research.gpuq run --label tm-holdout -- .venv/bin/python -m research.train_seg ...
    python -m research.gpuq run --label coco-v3 --no-gpu --writes outputs/coco_augmented_v3 -- .venv/bin/python ...
    python -m research.gpuq status

  --gpu (default)   exclusive local GPU; also waits until no unqueued process is computing on it
  --writes PATH     exclusive lock on PATH (waits for its readers and writers)
  --reads PATH      shared lock on PATH (many readers, no writer)
Locks are taken in a fixed order (paths sorted, GPU last), so two jobs can never deadlock. Waiting jobs are served
first come, first served. State lives in $WMLOC_QUEUE_DIR (default ~/.cache/wmloc-queue)."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("WMLOC_QUEUE_DIR", Path.home() / ".cache" / "wmloc-queue"))
POLL_S = 10


def _lock_file(key: str) -> Path:
    return ROOT / "locks" / (hashlib.sha1(key.encode()).hexdigest()[:16] + ".lock")


def _write_entry(path: Path, entry: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(entry))
    os.replace(tmp, path)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def entries() -> list[tuple[Path, dict]]:
    """Queue entries of live processes, oldest first; entries of dead processes are removed."""
    out = []
    for p in sorted((ROOT / "entries").glob("*.json")):
        try:
            e = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if _alive(e["pid"]):
            out.append((p, e))
        else:
            p.unlink(missing_ok=True)
    return out


def foreign_gpu_pids(own: set[int]) -> list[int]:
    """PIDs computing on the GPU that are not this job (unqueued work started outside the queue)."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [int(x) for x in r.stdout.split() if x.strip().isdigit() and int(x) not in own]


def conflicts(a: dict, b: dict) -> bool:
    """Two entries conflict when both need the GPU or one writes a path the other reads or writes."""
    keys_a, keys_b = set(a["writes"]) | set(a["reads"]), set(b["writes"]) | set(b["reads"])
    return (a["gpu"] and b["gpu"]) or bool(set(a["writes"]) & keys_b) or bool(set(b["writes"]) & keys_a)


def _wait_turn(me: Path) -> None:
    """First come, first served: wait until every older conflicting entry has finished."""
    mine = json.loads(me.read_text())
    while blocking := [e["label"] for p, e in entries() if p.name < me.name and conflicts(mine, e)]:
        _update(me, state=f"waiting for {', '.join(blocking)}")
        time.sleep(POLL_S)


def _update(path: Path, **kw) -> None:
    e = json.loads(path.read_text()); e.update(kw); _write_entry(path, e)


def run(a: argparse.Namespace) -> int:
    for d in ("locks", "entries", "logs"):
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    writes = sorted({str(Path(p).resolve()) for p in a.writes})
    reads = sorted({str(Path(p).resolve()) for p in a.reads} - set(writes))
    me = ROOT / "entries" / f"{time.time():.6f}-{os.getpid()}.json"
    entry = {"label": a.label, "pid": os.getpid(), "cwd": os.getcwd(), "cmd": a.cmd, "gpu": a.gpu,
             "writes": writes, "reads": reads, "queued_at": time.strftime("%H:%M:%S"), "state": "queued"}
    _write_entry(me, entry)
    held = []
    try:
        _wait_turn(me)
        plan = [(k, fcntl.LOCK_EX) for k in writes] + [(k, fcntl.LOCK_SH) for k in reads]
        plan.sort()
        if a.gpu:
            plan.append(("gpu", fcntl.LOCK_EX))
        for key, mode in plan:
            _update(me, state=f"locking {key}")
            f = open(_lock_file(key), "a+")
            fcntl.flock(f, mode)
            held.append(f)
        if a.gpu:
            while pids := foreign_gpu_pids({os.getpid()}):
                _update(me, state=f"waiting for unqueued GPU process {pids}")
                time.sleep(POLL_S)
        _update(me, state="running", started_at=time.strftime("%H:%M:%S"))
        print(f"[gpuq] {a.label}: running", file=sys.stderr, flush=True)
        return subprocess.call(a.cmd)
    finally:
        for f in held:
            f.close()
        me.unlink(missing_ok=True)


def status() -> None:
    es = entries()
    if not es:
        print("queue empty")
    for _, e in es:
        need = (["gpu"] if e["gpu"] else []) + [f"w:{p}" for p in e["writes"]] + [f"r:{p}" for p in e["reads"]]
        print(f"{e['queued_at']} pid {e['pid']:>7} {e['label']:<24} {e['state']:<40} {' '.join(need)}")
    others = foreign_gpu_pids({e["pid"] for _, e in es})
    if others:
        print(f"GPU processes outside the queue (or children of queued jobs): {others}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="action", required=True)
    r = sub.add_parser("run")
    r.add_argument("--label", required=True)
    r.add_argument("--no-gpu", dest="gpu", action="store_false")
    r.add_argument("--writes", action="append", default=[])
    r.add_argument("--reads", action="append", default=[])
    r.add_argument("cmd", nargs=argparse.REMAINDER)
    sub.add_parser("status")
    a = ap.parse_args()
    if a.action == "status":
        status(); return
    if a.cmd and a.cmd[0] == "--":
        a.cmd = a.cmd[1:]
    if not a.cmd:
        ap.error("run needs a command after --")
    sys.exit(run(a))


if __name__ == "__main__":
    main()
