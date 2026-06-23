"""Two single-message figures, one per headline conclusion.

C1  fig_c1_fusion_depth:   fusion-depth box-whiskers only (10 models).
    Message: native does not mean early fusion — fusion is mid-stack with or
    without an encoder; the one early case (LLaVA-1.5) is recipe, not class.

C2  fig_c2_vision_building: image-stream work strips only (10 models) +
    pre-Buffer markers + (where measured) the causal freeze check.
    Message: visual features are built once, unimodally, wherever the
    capacity lives — encoder, pre-Buffer, or early LLM layers.

Style matches fig_dol_pretty (Gillius ADF). Run with neo venv python.
"""

import glob
import json
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "neo_analysis"
OUT = Path(__file__).resolve().parents[1] / "neo_report"
INK = "#2b2b2b"
C_NAT, C_ENC = "#1f4e9c", "#e07b39"

MODELS = [
    ("NEO 1.0 · 2B", "native", "results_mat_neo_2B-SFT.jsonl", "results_fusion_full_neo_strat.jsonl", 12 / 40, "results_freeze_neo2b.jsonl"),
    ("NEO 1.0 · 9B", "native", "results_mat_neo_9B-SFT.jsonl", "results_fusion_full_neo_NEO1_0-9B-SFT.jsonl", 6 / 42, None),
    ("NEO 1.5 · 2B", "native", "results_mat_neo_15-2B-SFT.jsonl", "results_fusion_full_neo_NEO1_5-2B-SFT.jsonl", 12 / 40, None),
    ("NEO 1.5 · 9B", "native", "results_mat_neo_15-9B-SFT.jsonl", "results_fusion_full_neo_NEO1_5-9B-SFT.jsonl", 6 / 42, None),
    ("Gemma 4 · 12B", "native", "results_mat_gemma.jsonl", "results_fusion_full_gemma4_strat.jsonl", None, "results_freeze_gemma.jsonl"),
    ("SAIL · 7B", "native", "results_mat_sail.jsonl", "results_fusion_full_sail.jsonl", None, "results_freeze_sail.jsonl"),
    ("LLaVA 1.5 · 7B", "encoder", "results_mat_llava.jsonl", "results_fusion_full_llava.jsonl", None, "results_freeze_llava.jsonl"),
    ("LLaVA NeXT · 7B", "encoder", "results_mat_llavanext.jsonl", "results_win_llavanext.jsonl", None, None),
    ("OneVision · 7B", "encoder", "results_mat_onevision.jsonl", "results_win_onevision.jsonl", None, None),
    ("Qwen 2.5 VL · 7B", "encoder", "results_mat_qwen.jsonl", "results_fusion_full_qwenvl.jsonl", None, "results_freeze_qwenvl.jsonl"),
]


def setup_font():
    from matplotlib import font_manager, rcParams
    for f in glob.glob("/gscratch/krishna/leoym/fonts/gillius_ctan/gillius/opentype/GilliusADF-*.otf"):
        if "Cond" not in f and "Italic" not in f:
            font_manager.fontManager.addfont(f)
    rcParams["font.family"] = "Gillius ADF"
    rcParams["text.color"] = INK
    rcParams["axes.edgecolor"] = INK
    rcParams["xtick.color"] = INK


def load(path):
    p = ROOT / path
    if not p or not p.exists() or p.stat().st_size == 0:
        return None
    return [json.loads(l) for l in open(p) if l.strip() and '"skip"' not in l]


def fusion_quants(rows):
    causal = [r for r in rows if "cost" in r]
    N = len(causal[0]["cost"])

    def acc(g):
        return mean(g(r)["pred"] == r["gt"] for r in causal)
    R0 = acc(lambda r: r["intact"]) - acc(lambda r: r["swap"])
    rn = [(acc(lambda r, d=d: r["cost"][d]) - acc(lambda r, d=d: r["cost_null"][d])) / R0
          for d in range(N)]
    marg = [max((1 if d == 0 else rn[d - 1]) - rn[d], 0) for d in range(N)]
    tot = sum(marg)
    cdf, s = [], 0.0
    for m in marg:
        s += m
        cdf.append(s / tot)

    def q(p):
        return next(((i + 1) / N for i in range(N) if cdf[i] >= p), 1.0)
    return {k: q(v) for k, v in (("q10", .1), ("q25", .25), ("q50", .5), ("q75", .75), ("q90", .9))}


def u_img(rows):
    rows = [r for r in rows if "u_img" in r]
    N = max(len(r["u_img"]) for r in rows)
    rows = [r for r in rows if len(r["u_img"]) == N]
    return [mean(r["u_img"][l] for r in rows) for l in range(N)], N


def freeze_retention(rows, N):
    causal = [r for r in rows if "freeze_img" in r]
    ks = causal[0]["ks"]
    causal = [r for r in causal if r["ks"] == ks]

    def acc(g):
        return mean(g(r)["pred"] == r["gt"] for r in causal)
    R0 = acc(lambda r: r["intact"]) - acc(lambda r: r["swap"])
    j = min(range(len(ks)), key=lambda j: abs(ks[j] / N - 0.2))
    r0k = acc(lambda r: r["freeze_img"][j]) - acc(lambda r: r["freeze_img_null"][j])
    return max(r0k, 0) / R0


def style_axis(ax):
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1"], fontsize=10)
    for xv in (0.25, 0.5, 0.75):
        ax.axvline(xv, color=INK, lw=0.4, alpha=0.12, zorder=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("relative depth  ℓ / N", fontsize=11.5)


def fig_c1():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    y = 0.0
    yticks, ylabels = [], []
    last_fam = None
    for name, fam, _mf, ff, _pb, _fr in MODELS:
        fq = fusion_quants(load(ff))
        if last_fam and fam != last_fam:
            y -= 0.55
        last_fam = fam
        c = C_NAT if fam == "native" else C_ENC
        ax.plot([fq["q10"], fq["q90"]], [y, y], color=c, lw=1.3, zorder=3)
        for w in ("q10", "q90"):
            ax.plot([fq[w], fq[w]], [y - 0.14, y + 0.14], color=c, lw=1.3, zorder=3)
        ax.add_patch(plt.Rectangle((fq["q25"], y - 0.21), fq["q75"] - fq["q25"], 0.42,
                                   facecolor=c, alpha=0.18, edgecolor=c, lw=1.5, zorder=4))
        ax.plot(fq["q50"], y, "o", ms=9, mfc="white", mec=c, mew=1.9, zorder=6)
        yticks.append(y)
        ylabels.append(name)
        y -= 1.0
    ax.axvspan(1 / 3, 2 / 3, color=INK, alpha=0.045, zorder=0)
    ax.text(0.5, 0.78, "mid-stack", ha="center", fontsize=10, color=INK, alpha=0.65)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=11)
    ax.set_ylim(y + 0.4, 1.05)
    style_axis(ax)
    # legend
    h = [plt.Line2D([], [], color=C_NAT, marker="o", mfc="white", mew=1.8, lw=1.5, label="native (no encoder)"),
         plt.Line2D([], [], color=C_ENC, marker="o", mfc="white", mew=1.8, lw=1.5, label="encoder-based")]
    ax.legend(handles=h, frameon=False, fontsize=10, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_c1_fusion_depth.png", dpi=170)
    fig.savefig(OUT / "fig_c1_fusion_depth.pdf")
    print("wrote fig_c1_fusion_depth")


def fig_c2():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    H = 0.62
    y = 0.0
    yticks, ylabels = [], []
    last_fam = None
    for name, fam, mf, _ff, pb, fr in MODELS:
        ui, N = u_img(load(mf))
        if last_fam and fam != last_fam:
            y -= 0.55
        last_fam = fam
        base = mean(ui[1:]) or 1e-9
        v = np.clip(np.array(ui) / base, 0, 2.2) / 2.2
        x = np.linspace(0, 1, N + 1)
        for j in range(N):
            ax.add_patch(Rectangle((x[j], y - H / 2), x[j + 1] - x[j], H,
                                   color=plt.cm.Reds(0.08 + 0.86 * v[j]), lw=0))
        if pb:
            ax.plot([pb, pb], [y - H / 2 - 0.06, y + H / 2 + 0.06], color="white",
                    lw=2.6, zorder=4, solid_capstyle="butt")
            ax.plot([pb, pb], [y - H / 2 - 0.06, y + H / 2 + 0.06], color=INK,
                    lw=1.1, ls=(0, (2.6, 1.8)), zorder=5)
            ax.plot(pb, y + H / 2 + 0.12, marker="v", ms=4.5, color=INK, zorder=5)
        if fr:
            ret = freeze_retention(load(fr), N)
            ax.text(1.03, y, f"{ret:.2f}", fontsize=10.5, va="center",
                    color=("#1a7a3a" if ret >= 0.6 else "#b03030"),
                    fontweight="bold")
        yticks.append(y)
        ylabels.append(name)
        y -= 1.0
    ax.text(1.03, 1.05, "image signal kept\nwhen image stream\nis frozen (first 20%)",
            fontsize=8.5, va="bottom", ha="left", color=INK)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=11)
    ax.set_ylim(y + 0.4, 1.05)
    style_axis(ax)
    # legend: gradient + prebuffer marker
    fig.subplots_adjust(left=0.18, bottom=0.2, right=0.86, top=0.93)
    grad = np.linspace(0, 1, 128)[None, :]
    a = fig.add_axes([0.18, 0.05, 0.10, 0.02])
    a.imshow(grad, aspect="auto", cmap=plt.cm.Reds, vmin=-0.1, vmax=1.15)
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_lw(0.4)
    fig.text(0.29, 0.06, "image-stream work per layer", fontsize=9.5, va="center")
    g = fig.add_axes([0.56, 0.045, 0.018, 0.035])
    g.set_xlim(0, 1); g.set_ylim(0, 1)
    g.plot([0.5, 0.5], [0.05, 0.8], color=INK, lw=1.1, ls=(0, (2.6, 1.8)))
    g.plot(0.5, 0.9, marker="v", ms=4.5, color=INK)
    g.axis("off")
    fig.text(0.585, 0.06, "NEO pre-Buffer boundary", fontsize=9.5, va="center")
    fig.savefig(OUT / "fig_c2_vision_building.png", dpi=170)
    fig.savefig(OUT / "fig_c2_vision_building.pdf")
    print("wrote fig_c2_vision_building")


def main():
    import matplotlib
    matplotlib.use("Agg")
    setup_font()
    fig_c1()
    fig_c2()


if __name__ == "__main__":
    main()
