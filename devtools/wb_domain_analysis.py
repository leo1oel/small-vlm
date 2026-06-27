"""By-domain fusion-depth analysis for WorldBench activation-patch results.

WorldBench spans 7 visually-distinct domains (Living Things, Objects, Scenes, Digital
World, Academics, Documents/Charts/Tables, Agents). This computes the sufmeanabl
read-onset quantile (the fusion-depth metric, identical method to patch_analysis.py)
OVERALL and PER DOMAIN — the key test of whether dense-visual domains (Documents/Charts,
dense OCR) fuse at a different depth than coarse-object domains (Objects/Scenes).

Usage: python devtools/wb_domain_analysis.py <results_wb.jsonl> <tag> [min_domain_n]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_analysis import quants_from_retained  # noqa: E402


def load(path):
    return [json.loads(l) for l in open(path) if l.strip() and '"skip"' not in l]


def suf_quants(rows):
    """Overall sufmeanabl q50 + R0 + accuracies for a set of causal rows."""
    sm = [r for r in rows if "sufmeanabl" in r]
    if len(sm) < 4:
        return None
    iacc = mean(r["intact"]["pred"] == r["gt"] for r in sm)
    sacc = mean(r["swap"]["pred"] == r["gt"] for r in sm)
    R0 = iacc - sacc
    if abs(R0) < 1e-6:
        return None
    Nd = len(sm[0]["depths"]) - 1
    ret = [
        (
            mean(r["sufmeanabl"][d]["pred"] == r["gt"] for r in sm)
            - mean(r["sufmeanabl_null"][d]["pred"] == r["gt"] for r in sm)
        )
        / R0
        for d in range(Nd + 1)
    ]
    q = quants_from_retained(ret, rising=True)
    return {
        "n": len(sm),
        "intact": round(iacc, 3),
        "swap": round(sacc, 3),
        "R0": round(R0, 3),
        **{k: round(v, 3) for k, v in q.items()},
    }


def main():
    path, tag = sys.argv[1], sys.argv[2]
    min_n = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    rows = load(path)
    cz = [r for r in rows if "sufmeanabl" in r]
    overall = suf_quants(cz)
    print(f"== {tag} (WorldBench) ==  n_causal={len(cz)}")
    if overall:
        print(
            f"  OVERALL  intact={overall['intact']} swap={overall['swap']} R0={overall['R0']} "
            f"sufmeanabl q50={overall['q50']} [{overall['q25']},{overall['q75']}]"
            + ("  ⚠R0<0.05" if overall["R0"] < 0.05 else "")
        )
    by = defaultdict(list)
    for r in cz:
        by[r.get("domain", "?")].append(r)
    print(f"  -- by domain (min_n={min_n}) --")
    rowsout = []
    for dom in sorted(by):
        q = suf_quants(by[dom])
        if q is None or q["n"] < min_n:
            print(f"    {dom:28s} n={len(by[dom]):3d}  (too few / R0~0, skipped)")
            continue
        flag = "  ⚠R0<0.05" if q["R0"] < 0.05 else ""
        print(
            f"    {dom:28s} n={q['n']:3d}  R0={q['R0']:+.2f}  q50={q['q50']:.2f} [{q['q25']:.2f},{q['q75']:.2f}]{flag}"
        )
        rowsout.append((dom, q))
    # spread across domains (the headline number: does fusion depth move with visual domain?)
    qs = [q["q50"] for _, q in rowsout if q["R0"] >= 0.05]
    if len(qs) >= 2:
        print(
            f"  => domain q50 spread: min={min(qs):.2f} max={max(qs):.2f} range={max(qs) - min(qs):.2f} "
            f"(small range = fusion depth is domain-invariant)"
        )
    out = {"tag": tag, "overall": overall, "by_domain": {d: q for d, q in rowsout}}
    Path(f"neo_analysis/wb_domain_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"  saved neo_analysis/wb_domain_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
