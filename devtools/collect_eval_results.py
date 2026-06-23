"""Collect all lmms-eval results under logs/lmms_eval into one tidy table.

Usage: python devtools/collect_eval_results.py [--csv out.csv]

Walks logs/lmms_eval/<run>__<checkpoint>/<ts>_results.json (latest per task
wins), extracts every numeric metric, and prints a per-benchmark pivot:
rows = (run, step), columns = metrics.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "logs" / "lmms_eval"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="also write a flat CSV here")
    args = ap.parse_args()

    # (run, step, task) -> {metric: value}; later timestamps overwrite earlier.
    table: dict[tuple, dict] = defaultdict(dict)
    for results_file in sorted(ROOT.glob("*/*_results.json")):
        m = re.match(r"(.+)__checkpoint-(\d+)$", results_file.parent.name)
        if not m:
            continue
        run, step = m.group(1), int(m.group(2))
        try:
            data = json.loads(results_file.read_text())
        except Exception:
            continue
        for task, metrics in (data.get("results") or {}).items():
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and not k.endswith("_stderr"):
                    name = k.split(",")[0]
                    if name == "alias":
                        continue
                    table[(run, step, task)][name] = v

    tasks = sorted({t for (_, _, t) in table})
    rows = []
    for task in tasks:
        keys = sorted([k for k in table if k[2] == task])
        metrics = sorted({m for k in keys for m in table[k]})
        print(f"\n## {task}")
        header = f"{'run':<34} {'step':>6} | " + " ".join(f"{m[:16]:>16}" for m in metrics)
        print(header)
        print("-" * len(header))
        for run, step, _ in keys:
            vals = table[(run, step, task)]
            cells = " ".join(
                f"{vals[m] * 100:>16.2f}" if m in vals else f"{'-':>16}" for m in metrics
            )
            print(f"{run:<34} {step:>6} | {cells}")
            rows.append({"run": run, "step": step, "task": task, **vals})

    if args.csv:
        import csv

        fields = sorted({k for r in rows for k in r})
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
