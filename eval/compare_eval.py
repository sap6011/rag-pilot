"""Compare the two most recent eval runs side by side."""
import json
from pathlib import Path

runs = sorted(Path("eval/runs").glob("run_*.json"))
if len(runs) < 2:
    print(f"Need at least 2 runs to compare. Found {len(runs)}.")
    raise SystemExit

prev = json.loads(runs[-2].read_text())
curr = json.loads(runs[-1].read_text())

print(f"Previous: {runs[-2].name}")
print(f"Current:  {runs[-1].name}\n")

def fmt(v):
    return f"{v:.3f}" if v is not None else "n/a"

p, c = prev["meta"]["summary"], curr["meta"]["summary"]
print(f"{'Metric':<22}{'Previous':>12}{'Current':>12}{'Δ':>12}")
print("-" * 58)
for key in ["recall_at_k", "mrr", "keyword_coverage"]:
    pv, cv = p.get(key), c.get(key)
    delta = (cv - pv) if (pv is not None and cv is not None) else None
    delta_str = f"{delta:+.3f}" if delta is not None else "n/a"
    print(f"{key:<22}{fmt(pv):>12}{fmt(cv):>12}{delta_str:>12}")