"""Companion causal summary: image-signal retention when one stream's in-LLM
processing is cancelled through the first ~20% and ~30% of layers.

retention = R0(freeze@k) / R0(baseline), k chosen nearest 0.2N / 0.3N.
Encoder models should retain ~1 under freeze_img (their LLM never needs to
process image tokens); natives should crash; freeze_txt crashes everyone
except NEO's pre-Buffer region.

Run with neo venv python. Usage: python devtools/fig_freeze_bars.py
"""

import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1] / "neo_analysis"
OUT = Path(__file__).resolve().parents[1] / "neo_report"

MODELS = [
    ("LLaVA-1.5-7B", "encoder", "results_freeze_llava.jsonl"),
    ("Qwen2.5-VL-7B", "encoder", "results_freeze_qwenvl.jsonl"),
    ("Gemma-4-12B", "native", "results_freeze_gemma.jsonl"),
    ("NEO1.0-2B", "native", "results_freeze_neo2b.jsonl"),
    ("SAIL-7B", "native", "results_freeze_sail.jsonl"),
]
NS = {"LLaVA-1.5-7B": 32, "Qwen2.5-VL-7B": 28, "Gemma-4-12B": 48, "NEO1.0-2B": 40, "SAIL-7B": 32}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    data = []
    for name, fam, f in MODELS:
        rows = [json.loads(l) for l in open(ROOT / f) if l.strip() and '"skip"' not in l]
        causal = [r for r in rows if "freeze_img" in r]
        ks = causal[0]["ks"]
        causal = [r for r in causal if r["ks"] == ks]
        N = NS[name]

        def acc(g):
            return mean(g(r)["pred"] == r["gt"] for r in causal)
        R0 = acc(lambda r: r["intact"]) - acc(lambda r: r["swap"])
        j2 = min(range(len(ks)), key=lambda j: abs(ks[j] / N - 0.2))
        rec = dict(name=name, fam=fam, R0=R0)
        for mode in ("img", "txt"):
            r0k = (acc(lambda r: r[f"freeze_{mode}"][j2]) -
                   acc(lambda r: r[f"freeze_{mode}_null"][j2]))
            rec[mode] = max(r0k, 0) / R0
        data.append(rec)
        print(f"{name:14} ({fam}) R0={R0:+.2f} | retention@~0.2N: freeze_img={rec['img']:.2f} freeze_txt={rec['txt']:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    x = np.arange(len(data))
    colors = ["#ff7f0e" if d["fam"] == "encoder" else "#1f77b4" for d in data]
    for ax, mode, title, sub in (
        (axes[0], "img", "Test 1: FREEZE the IMAGE stream (first 20% of layers)",
         "question: does the LLM itself need to process image tokens?"),
        (axes[1], "txt", "Test 2: FREEZE the TEXT stream (first 20% of layers)",
         "question: does the LLM need to process text tokens early?"),
    ):
        vals = [d[mode] for d in data]
        ax.bar(x, vals, 0.62, color=colors)
        ax.axhline(1.0, color="green", lw=1.2, ls="--")
        ax.text(len(data) - 0.4, 1.03, "1.0 = harmless\n(that processing was NOT needed)",
                fontsize=8, color="green", ha="right")
        ax.axhline(0.0, color="k", lw=0.8)
        ax.text(len(data) - 0.4, 0.04, "0 = image signal destroyed\n(that processing WAS needed)",
                fontsize=8, color="dimgray", ha="right")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{d['name'].replace('-7B','').replace('-12B','')}\n{d['fam']}"
                            for d in data], fontsize=9)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.04, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
        ax.set_title(title + "\n" + sub, fontsize=9.5)
        ax.set_ylim(0, 1.3)
    axes[0].set_ylabel("image signal kept  R0(frozen) / R0(normal)")
    fig.suptitle("Causal check of the labor map: cancel one stream's early in-LLM processing, "
                 "measure how much usable image signal survives", fontsize=10.5, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_freeze_bars.png", dpi=140, bbox_inches="tight")
    print("wrote fig_freeze_bars.png")


if __name__ == "__main__":
    main()
