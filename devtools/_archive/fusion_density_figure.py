"""Cleaner fusion-depth view: cumulative committed-signal curve C(d) and a
Fusion Depth Index (FDI) dot-chart. f(d) (the raw per-layer density) is a noisy
derivative; C(d) = 1 - retained(d)/R0 is its smooth integral and shows the
pattern directly, and the FDI dot-chart shows every config clustering at mid.

  f(d)  = (retained(d-1)-retained(d))/R0     per-layer fusion density (noisy)
  C(d)  = 1 - retained(d)/R0                 fraction of image signal already
                                             functionally committed by depth d
  FDI   = sum_d (d/N) f(d)                    early(<.33)/mid/late(>.66) scalar

Usage: /envs/neo/bin/python devtools/fusion_density_figure.py <out.png>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fusion_crossmodel_figure import curves, load  # noqa: E402

# (label, file, color, draw_curve?) — PT excluded (caption model, R0<0).
MODELS = [
    ("bee-mix 2B", "results_fusion_full_base5000.jsonl", "#1f77b4", True),
    ("NEO1.0-2B MT", "results_fusion_full_neo_MT.jsonl", "#8c564b", False),
    ("NEO1.0-2B SFT", "results_fusion_full_neo.jsonl", "#2ca02c", True),
    ("NEO1.0-9B SFT", "results_fusion_full_neo_NEO1_0-9B-SFT.jsonl", "#17becf", True),
    ("NEO1.5-2B SFT", "results_fusion_full_neo_NEO1_5-2B-SFT.jsonl", "#9467bd", False),
    ("NEO1.5-9B SFT", "results_fusion_full_neo_NEO1_5-9B-SFT.jsonl", "#e377c2", False),
    ("Gemma-4-12B", "results_fusion_full_gemma4.jsonl", "#d62728", True),
]


def fdi_of(c):
    N, R0, prev, raw = c["N"], c["R0"], c["R0"], []
    for d in range(N):
        raw.append(max(prev - c["retained"][d], 0.0)); prev = c["retained"][d]
    tot = sum(raw) or 1e-9
    return sum((d + 1) / N * raw[d] / tot for d in range(N))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = []
    for name, path, color, curve in MODELS:
        rows = load(path)
        if rows is None:
            print(f"  (missing {path})"); continue
        c = curves(rows)
        fdi = fdi_of(c)
        data.append((name, color, curve, c, fdi))
        print(f"{name:16s} N={c['N']:2d} R0={c['R0']:+.3f} FDI={fdi:.3f} "
              f"({'early' if fdi < .33 else 'mid' if fdi < .66 else 'late'})")

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5), gridspec_kw={"width_ratios": [1.4, 1]})

    # left: cumulative committed-signal curves (smooth) for the representative subset
    for name, color, curve, c, fdi in data:
        if not curve:
            continue
        x = [(i + 1) / c["N"] for i in range(c["N"])]
        cum = [max(0.0, min(1.0, 1 - c["retained"][i] / c["R0"])) for i in range(c["N"])]
        ax[0].plot(x, cum, "-", lw=2, color=color, label=f"{name} (FDI {fdi:.2f})")
    ax[0].axvspan(0.33, 0.66, color="gray", alpha=.10)
    ax[0].axhline(0.5, color="k", lw=.6, ls="--", alpha=.4)
    ax[0].set_title("Cumulative image signal functionally committed by depth d\n"
                    "(flat early = pre-fusion; rise = fusion zone)")
    ax[0].set_xlabel("relative depth d / N")
    ax[0].set_ylabel("fraction committed  C(d) = 1 − retained/R0")
    ax[0].set_ylim(-0.05, 1.05); ax[0].legend(fontsize=8, loc="upper left")

    # right: FDI dot-chart — every config, with early/mid/late bands
    ax[1].axvspan(0, 0.33, color="#fdd", alpha=.5)
    ax[1].axvspan(0.33, 0.66, color="#dfd", alpha=.5)
    ax[1].axvspan(0.66, 1.0, color="#ddf", alpha=.5)
    ys = list(range(len(data)))
    for y, (name, color, curve, c, fdi) in zip(ys, data):
        ax[1].scatter([fdi], [y], s=90, color=color, zorder=3, edgecolor="k", linewidth=.5)
        ax[1].annotate(f"{fdi:.2f}", (fdi, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax[1].set_yticks(ys); ax[1].set_yticklabels([d[0] for d in data], fontsize=8)
    ax[1].set_xlim(0, 1); ax[1].set_xlabel("Fusion Depth Index (FDI)")
    ax[1].invert_yaxis()
    ax[1].text(0.165, len(data) - 0.4, "early", ha="center", fontsize=8, color="#a00")
    ax[1].text(0.495, len(data) - 0.4, "mid", ha="center", fontsize=8, color="#0a0")
    ax[1].text(0.83, len(data) - 0.4, "late", ha="center", fontsize=8, color="#00a")
    ax[1].set_title("FDI per config — all cluster at mid (~0.5)")

    fig.suptitle("Where functional fusion lives: cumulative commitment + Fusion Depth Index, VMCBench dev", y=1.02)
    fig.tight_layout()
    fig.savefig(sys.argv[1], dpi=130, bbox_inches="tight")
    print(f"saved -> {sys.argv[1]}")


if __name__ == "__main__":
    main()
