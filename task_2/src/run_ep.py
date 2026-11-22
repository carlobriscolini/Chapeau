"""Send a prompt file to the SCORED endpoints ep-01..ep-33. Spends the real budget: run only with the user's OK.

    python run_ep.py ep_phase1.jsonl --out data/ep_phase1.jsonl --reserve 30 --yes-user-approved
    python run_ep.py ep_phase1.jsonl --dry-run        # only prints the plan and the current budget

1 answer per prompt and endpoint (a repeated prompt has its own id, e.g. "cont_physics#r2").
--reserve N refuses to start if the run would leave fewer than N answers on some endpoint.
Resume-safe (answers already in --out are skipped), round-robin over endpoints (official query()).
"""
import argparse
from pathlib import Path

import common
import query_reference as qr

ap = argparse.ArgumentParser()
ap.add_argument("prompts")
ap.add_argument("--out", default=None)
ap.add_argument("--endpoints", default=None, help="comma list; default all ep-*")
ap.add_argument("--reserve", type=int, default=30)
ap.add_argument("--max-tokens", type=int, default=256)
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--yes-user-approved", action="store_true")
a = ap.parse_args()

prompts = qr.read_prompts(Path(a.prompts))
out = Path(a.out) if a.out else common.DATA / f"ep_{Path(a.prompts).stem}.jsonl"
left = {e: v for e, v in common._get_budget_robust(common.session()).items() if e.startswith("ep-")}
eps = a.endpoints.split(",") if a.endpoints else sorted(left)
assert eps and all(e.startswith("ep-") for e in eps), eps
done = set()
if out.exists():
    done = {(r["endpoint"], r["id"]) for r in common.load_answers(out)}
todo = {e: sum((e, pid) not in done for pid, _ in prompts) for e in eps}
print(f"{len(prompts)} prompt x {len(eps)} endpoint -> {out}")
print(f"query da fare: {sum(todo.values())} (per endpoint: max {max(todo.values())})")
print("budget residuo ora: min", min(left[e] for e in eps), "max", max(left[e] for e in eps))
after = {e: left[e] - todo[e] for e in eps}
low = {e: v for e, v in after.items() if v < a.reserve}
print("budget residuo dopo: min", min(after.values()), f"(riserva richiesta {a.reserve})")
if low:
    raise SystemExit(f"STOP: lascerebbe meno di {a.reserve} risposte su {sorted(low)}")
if a.dry_run:
    raise SystemExit("dry-run: nessuna query inviata")
if not a.yes_user_approved:
    raise SystemExit("STOP: serve --yes-user-approved (ok esplicito dell'utente)")
common.query_safe(prompts, out, eps, repeats=1, allow_scored=True, max_tokens=a.max_tokens)
