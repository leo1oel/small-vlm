"""WorldBench fusion-depth figure: (1) cross-benchmark invariance — sufmeanabl q50 on
VMCBench vs MMStar vs WorldBench per model (should be flat → fusion depth is a model
property, not a benchmark artifact); (2) by-domain WorldBench q50 for the strongest
anchor (should be a tight band → fusion depth is visual-domain-invariant), with
no-signal (R0<0.05) domains greyed.

Reads neo_analysis/wb_domain_<tag>.json (from wb_domain_analysis.py). VMCBench/MMStar
baselines are from GENERALIZATION_RESULTS.md. Run with neo venv python (matplotlib).

Usage: python devtools/fig_worldbench.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "neo_analysis"
OUT = Path(__file__).resolve().parents[1] / "neo_report" / "fig_worldbench.png"

# baselines (sufmeanabl q50) from GENERALIZATION_RESULTS.md
BASE = {  # tag: (label, vmcbench, mmstar)
    "wb_ivl8": ("InternVL3.5-8B", 0.42, 0.47),
    "wb_gemma12": ("Gemma-4-12B", 0.42, None),
    "wb_qwen": ("Qwen2.5-VL-7B", 0.46, 0.50),
}


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ---- panel 1: cross-benchmark invariance ----
    tags = [t for t in BASE if (ROOT / f"wb_domain_{t}.json").exists()]
    x = np.arange(len(tags))
    w = 0.27
    vmc = [BASE[t][1] for t in tags]
    mms = [BASE[t][2] if BASE[t][2] is not None else np.nan for t in tags]
    wb = [json.load(open(ROOT / f"wb_domain_{t}.json"))["overall"]["q50"] for t in tags]
    ax1.bar(x - w, vmc, w, label="VMCBench", color="#8da0cb")
    ax1.bar(x, mms, w, label="MMStar", color="#66c2a5")
    ax1.bar(x + w, wb, w, label="WorldBench", color="#fc8d62")
    ax1.set_xticks(x)
    ax1.set_xticklabels([BASE[t][0] for t in tags], rotation=20, ha="right", fontsize=9)
    ax1.set_ylabel("fusion depth  (sufmeanabl q50, relative)")
    ax1.set_ylim(0, 0.7)
    ax1.axhspan(0.35, 0.55, color="#cccccc", alpha=0.3, zorder=0)  # mid-stack band
    ax1.set_title(
        "Fusion depth is benchmark-invariant\n(WorldBench tracks VMCBench/MMStar, all mid-stack)"
    )
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25, axis="y")

    # ---- panel 2: by-domain (strongest anchor) ----
    # by-domain panel: prefer the full-data anchor (Gemma ran all 2000; InternVL timed out at 875)
    anchor = (
        "wb_gemma12"
        if (ROOT / "wb_domain_wb_gemma12.json").exists()
        else (tags[0] if tags else None)
    )
    if anchor:
        d = json.load(open(ROOT / f"wb_domain_{anchor}.json"))
        doms = sorted(d["by_domain"].keys())
        q = [d["by_domain"][k]["q50"] for k in doms]
        r0 = [d["by_domain"][k]["R0"] for k in doms]
        cols = ["#fc8d62" if rr >= 0.05 else "#dddddd" for rr in r0]
        ax2.bar(range(len(doms)), q, color=cols)
        if d.get("overall"):
            ax2.axhline(
                d["overall"]["q50"],
                color="#d53e4f",
                ls="--",
                lw=1,
                label=f"overall q50={d['overall']['q50']:.2f}",
            )
        ax2.axhspan(0.35, 0.55, color="#cccccc", alpha=0.3, zorder=0)
        ax2.set_xticks(range(len(doms)))
        ax2.set_xticklabels(
            [k.replace(", ", ",\n").replace(" ", "\n", 1) for k in doms], fontsize=7
        )
        ax2.set_ylabel("fusion depth (sufmeanabl q50)")
        ax2.set_ylim(0, 0.7)
        ax2.set_title(
            f"{BASE.get(anchor, (anchor,))[0]}: by visual domain\n(grey = R0<0.05, no signal; "
            "coloured = mid-stack regardless of domain)"
        )
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.25, axis="y")

    plt.tight_layout()
    plt.savefig(OUT, dpi=140)
    print(f"[fig] saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
