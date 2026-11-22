"""Shared helpers: API key loading, budget check, safe querying.

The API key is read from the MET_API_KEY environment variable or from the file .api_key (never committed).
Querying reuses the official query() from query_reference.py (resume-safe, budget-aware, 4 parallel requests).

SAFETY: query_safe() refuses scored endpoints (ep-*) unless allow_scored=True is passed explicitly.
"""
import json
import os
from pathlib import Path

import requests

import query_reference as qr

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)


def api_key() -> str:
    key = os.environ.get("MET_API_KEY", "").strip()
    if not key and (ROOT / ".api_key").exists():
        key = (ROOT / ".api_key").read_text().strip()
    if not key:
        raise SystemExit("No API key: put it in .api_key or export MET_API_KEY=...")
    return key


def session() -> requests.Session:
    qr.API_KEY = api_key()
    return qr.make_session()


def models_raw() -> list:
    """Raw /models listing (free: no generation)."""
    r = session().get(f"{qr.BASE_URL}/models", timeout=60)
    r.raise_for_status()
    return r.json()["data"]


def budgets() -> dict:
    return {m["id"]: m.get("budget") for m in models_raw()}


def _get_budget_robust(sess) -> dict:
    """Like qr.get_budget but tolerates endpoints without a numeric limit (unlimited reference endpoints)."""
    reply = sess.get(f"{qr.BASE_URL}/models", timeout=60)
    reply.raise_for_status()
    out = {}
    for m in reply.json()["data"]:
        b = m.get("budget") or {}
        lim, used = b.get("limit"), b.get("used") or 0
        out[m["id"]] = 10**9 if lim is None else lim - used
    return out


qr.get_budget = _get_budget_robust


def query_safe(prompts: list, output: Path, endpoints: list, repeats: int = 1, allow_scored: bool = False,
               max_tokens: int = 256) -> None:
    scored = [e for e in endpoints if not e.startswith("ref-")]
    if scored and not allow_scored:
        raise SystemExit(f"Refusing to query scored endpoints {scored} without allow_scored=True")
    session()  # sets qr.API_KEY
    qr.query(prompts, output=Path(output), endpoints=endpoints, repeats=repeats, max_tokens=max_tokens)


def load_answers(path: Path) -> list:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
