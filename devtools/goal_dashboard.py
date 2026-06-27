"""Dashboard for the 1000-step encoder-free catch-up goal (2026-06-22).
Tabulates vmcbench/pope/mmvp across the goal arms and the gap to the CLIP-encoder
bar (sft-clip-bee-mix) at matching steps. The goal: a native arm within 5
vmcbench points of the bar at step 1000.

  python devtools/goal_dashboard.py
"""

import glob
import json
import os
import re

ARMS = [
    ("sft-clip-bee-mix", "R0 CLIP-encoder BAR"),
    ("sft-unified-bee-mix-warmstem", "W1 warmstem+delta"),
    ("sft-unified-bee-mix-warmstem-distill", "W2 warmstem+distill+delta"),
    ("sft-unified-bee-mix-visualffn-1k", "W3 visualffn+delta (BREEN)"),
    ("sft-unified-bee-mix-warmstem-visualffn", "W4 warmstem+visualffn+delta"),
    ("sft-unified-bee-mix-visualffn-distill", "C1 visualffn+distill (joint)"),
    ("sft-unified-bee-mix-prefix-distill", "C2 prefix+distill (joint)"),
    ("sft-unified-bee-mix-warmstem-joint", "W5 warmstem+JOINT (no delta)"),
    ("sft-unified-bee-mix-warmstem-mlp-distill", "G2 warmstem+MLP+distill (1000s)"),
    ("sft-unified-bee-mix", "native baseline (ref)"),
]


def metrics(f):
    try:
        d = json.load(open(f))
    except Exception:
        return {}
    r = d.get("results", {})
    out = {}
    v = r.get("vmcbench", {}).get("average,none")
    if isinstance(v, (int, float)):
        out["vmc"] = round(v * 100, 1)
    p = r.get("pope", {}).get("pope_accuracy,none") or r.get("pope", {}).get("f1,none")
    if isinstance(p, (int, float)):
        out["pope"] = round(p * 100, 1)
    m = r.get("mmvp", {}).get("mmvp_accuracy,none")
    if isinstance(m, (int, float)):
        out["mmvp"] = round(m * 100, 1)
    return out


def collect():
    # arm -> step -> {metric: best}
    data = {}
    for f in glob.glob("logs/lmms_eval/*/*_results.json"):
        name = f.split("/")[-2]
        mm = re.search(r"^(.*)__checkpoint-(\d+)$", name)
        if not mm:
            continue
        arm, step = mm.group(1), int(mm.group(2))
        met = metrics(f)
        d = data.setdefault(arm, {}).setdefault(step, {})
        for k, val in met.items():
            if k not in d or val > d[k]:  # best across reruns
                d[k] = val
    return data


def main():
    os.chdir("/mmfs1/gscratch/krishna/leoym/small-vlm")
    data = collect()
    bar = data.get("sft-clip-bee-mix", {})

    def barvmc(step):
        # nearest bar checkpoint <= step (the bar at the same budget)
        cand = [s for s in bar if "vmc" in bar[s]]
        if not cand:
            return None
        near = min(cand, key=lambda s: abs(s - step))
        return bar[near].get("vmc"), near

    print(f"\n{'arm':32s} {'step':>5} {'vmc':>5} {'pope':>5} {'mmvp':>5} {'gap→bar':>8}")
    print("-" * 72)
    for arm, label in ARMS:
        steps = sorted(data.get(arm, {}))
        if not steps:
            print(f"{label:32s}  (no eval yet)")
            continue
        for s in steps:
            m = data[arm][s]
            vmc = m.get("vmc")
            gap = ""
            if vmc is not None and arm != "sft-clip-bee-mix":
                bv = barvmc(s)
                if bv and bv[0] is not None:
                    gap = f"{vmc - bv[0]:+.1f}@{bv[1]}"
            print(
                f"{label:32s} {s:>5} {str(vmc or ''):>5} {str(m.get('pope') or ''):>5} "
                f"{str(m.get('mmvp') or ''):>5} {gap:>8}"
            )
    print(
        "\nGOAL: a native arm with gap ≥ -5.0 (within 5 vmcbench points of the bar) at step 1000."
    )


if __name__ == "__main__":
    main()
