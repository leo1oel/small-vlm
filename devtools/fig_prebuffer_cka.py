"""Figure: NEO's pre-Buffer builds vision-encoder-grade features.

Linear CKA between each model's per-layer mean-pooled image-token
representations and a frozen reference vision encoder (DINOv2, never used by
any of these models), over relative depth. NEO curves climb steeply inside
the (shaded) pre-Buffer and peak at encoder-output level — the level at which
Qwen2.5-VL's real ViT delivers its features into the LLM (entry point, dotted
line). SAIL (native, no pre-Buffer) shows no build-up phase.

Style matches fig_dol_pretty (Gillius ADF). Usage: neo venv python.
"""

import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "neo_analysis"
OUT = Path(__file__).resolve().parents[1] / "neo_report"
INK = "#2b2b2b"


def setup_font():
    from matplotlib import font_manager, rcParams
    for f in glob.glob("/gscratch/krishna/leoym/fonts/gillius_ctan/gillius/opentype/GilliusADF-*.otf"):
        if "Cond" not in f and "Italic" not in f:
            font_manager.fontManager.addfont(f)
    rcParams["font.family"] = "Gillius ADF"
    rcParams["text.color"] = INK
    rcParams["axes.edgecolor"] = INK
    rcParams["xtick.color"] = INK
    rcParams["ytick.color"] = INK


def main():
    import matplotlib
    matplotlib.use("Agg")
    setup_font()
    import matplotlib.pyplot as plt

    d = json.loads((ROOT / "cka_results.json").read_text())
    REF = "DINOv2"

    series = [
        ("NEO 1.0 · 2B", "NEO1.0-2B-SFT", "#1f4e9c", "-", 12 / 40),
        ("NEO 1.0 · 9B", "NEO1.0-9B-SFT", "#2e8bc0", "-", 6 / 42),
        ("SAIL · 7B  (native, no pre-Buffer)", "SAIL-7B", "#8a8a8a", (0, (4, 2)), None),
        ("Qwen 2.5 VL · 7B  (encoder-based)", "Qwen2.5-VL-7B", "#e07b39", (0, (1.5, 1.5)), None),
    ]

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    qwen0 = d["Qwen2.5-VL-7B"][REF][0]
    ax.axhline(qwen0, color="#e07b39", lw=1.0, ls=":", alpha=0.9, zorder=1)
    ax.text(0.995, qwen0 + 0.015,
            "output level of a real vision encoder\n(Qwen ViT features at LLM entry)",
            ha="right", va="bottom", fontsize=9, color="#b65c1e")

    for label, key, color, ls, pb in series:
        c = d[key][REF]
        L = len(c)
        x = [i / (L - 1) for i in range(L)]
        ax.plot(x, c, color=color, ls=ls, lw=2.2 if pb else 1.7,
                label=label, zorder=3, solid_capstyle="round")
        if pb:
            ax.axvspan(0, pb, color=color, alpha=0.08, zorder=0)
            ax.plot([pb, pb], [0.05, 0.8], color=color, lw=1.0,
                    ls=(0, (2.6, 1.8)), zorder=2)
            ax.plot(pb, 0.8, marker="v", ms=5, color=color, zorder=4)
        # entry marker
        ax.plot(x[0], c[0], "o", ms=6, mfc="white", mec=color, mew=1.6, zorder=4)

    ax.text(12 / 40 + 0.012, 0.115, "pre-Buffer\nboundary (2B)", fontsize=8.5,
            color="#1f4e9c", va="bottom")
    ax.text(6 / 42 + 0.012, 0.185, "pre-Buffer\nboundary (9B)", fontsize=8.5,
            color="#2e8bc0", va="bottom")

    ax.set_xlim(0, 1)
    ax.set_ylim(0.05, 0.88)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1"], fontsize=10)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.tick_params(axis="y", labelsize=10, length=0)
    ax.tick_params(axis="x", length=3)
    for xv in (0.25, 0.5, 0.75):
        ax.axvline(xv, color=INK, lw=0.4, alpha=0.12, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.8)
    ax.set_xlabel("relative depth  ℓ / N", fontsize=11.5)
    ax.set_ylabel(f"CKA( image-token representations,  {REF} )", fontsize=11.5)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right", handlelength=2.6)
    fig.tight_layout()
    fig.savefig(OUT / "fig_prebuffer_cka.png", dpi=170)
    fig.savefig(OUT / "fig_prebuffer_cka.pdf")
    print("wrote fig_prebuffer_cka.{png,pdf}")


if __name__ == "__main__":
    main()
