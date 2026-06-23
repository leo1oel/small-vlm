"""Polished Division-of-Labor figure (publication style).

6 native + 4 encoder VLMs (MT variants dropped), no header, Gillius ADF
(open Gill Sans). Strips: image-stream / text-stream per-layer work; black
box-whisker: fusion-depth distribution (q10-q25-median-q75-q90) from the
prefix attention-knockout necessity profile.

Run with neo venv python. Usage: python devtools/fig_dol_pretty.py
"""

import glob
import json
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "neo_analysis"
OUT = Path(__file__).resolve().parents[1] / "neo_report"

MODELS = [
    ("NEO 1.0 · 2B", "native", "results_mat_neo_2B-SFT.jsonl", "results_fusion_full_neo_strat.jsonl", 12 / 40),
    ("NEO 1.0 · 9B", "native", "results_mat_neo_9B-SFT.jsonl", "results_fusion_full_neo_NEO1_0-9B-SFT.jsonl", 6 / 42),
    ("NEO 1.5 · 2B", "native", "results_mat_neo_15-2B-SFT.jsonl", "results_fusion_full_neo_NEO1_5-2B-SFT.jsonl", 12 / 40),
    ("NEO 1.5 · 9B", "native", "results_mat_neo_15-9B-SFT.jsonl", "results_fusion_full_neo_NEO1_5-9B-SFT.jsonl", 6 / 42),
    ("Gemma 4 · 12B", "native", "results_mat_gemma.jsonl", "results_fusion_full_gemma4_strat.jsonl", None),
    ("SAIL · 7B", "native", "results_mat_sail.jsonl", "results_fusion_full_sail.jsonl", None),
    ("LLaVA 1.5 · 7B", "encoder", "results_mat_llava.jsonl", "results_fusion_full_llava.jsonl", None),
    ("LLaVA NeXT · 7B", "encoder", "results_mat_llavanext.jsonl", "results_win_llavanext.jsonl", None),
    ("OneVision · 7B", "encoder", "results_mat_onevision.jsonl", "results_win_onevision.jsonl", None),
    ("Qwen 2.5 VL · 7B", "encoder", "results_mat_qwen.jsonl", "results_fusion_full_qwenvl.jsonl", None),
]

INK = "#2b2b2b"


def setup_font():
    from matplotlib import font_manager, rcParams
    added = False
    for f in glob.glob("/gscratch/krishna/leoym/fonts/gillius_ctan/gillius/opentype/GilliusADF-*.otf"):
        if "Cond" not in f and "Italic" not in f:
            font_manager.fontManager.addfont(f)
            added = True
    for f in glob.glob("/gscratch/krishna/leoym/fonts/gillius_ctan/gillius/opentype/GilliusADF-Bold.otf"):
        font_manager.fontManager.addfont(f)
    if added:
        rcParams["font.family"] = "Gillius ADF"
    rcParams["text.color"] = INK
    rcParams["axes.edgecolor"] = INK
    rcParams["xtick.color"] = INK


def load(path):
    p = ROOT / path
    if not p.exists() or p.stat().st_size == 0:
        return None
    return [json.loads(l) for l in open(p) if l.strip() and '"skip"' not in l]


def mat_curves(rows):
    rows = [r for r in rows if "u_img" in r]
    N = max(len(r["u_img"]) for r in rows)
    rows = [r for r in rows if len(r["u_img"]) == N]
    ui = [mean(r["u_img"][l] for r in rows) for l in range(N)]
    ut = [mean(r["u_txt"][l] for r in rows) for l in range(N)]
    return ui, ut, N


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
    return {p: q(v) for p, v in (("q10", .10), ("q25", .25), ("q50", .50),
                                 ("q75", .75), ("q90", .90))}


def strip_vals(u, clip=2.2):
    base = mean(u[1:]) or 1e-9
    return np.clip(np.array(u) / base, 0, clip) / clip  # -> [0,1]


def main():
    import matplotlib
    matplotlib.use("Agg")
    setup_font()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    cm_img = plt.cm.Reds
    cm_txt = plt.cm.Purples

    rows = []
    for name, fam, mf, ff, pb in MODELS:
        m, f = load(mf), load(ff)
        if not (m and f):
            print("missing", name)
            continue
        rows.append((name, fam, mat_curves(m), fusion_quants(f), pb))

    H, GAP, GROUP_GAP = 0.62, 0.38, 0.65
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    y = 0.0
    yticks, ylabels = [], []
    last_fam = None
    group_anchor = {}
    for name, fam, (ui, ut, N), fq, pb in rows:
        if last_fam and fam != last_fam:
            y -= GROUP_GAP
        last_fam = fam
        group_anchor.setdefault(fam, []).append(y)
        x = np.linspace(0, 1, N + 1)
        si, st = strip_vals(ui), strip_vals(ut)
        for j in range(N):
            ax.add_patch(Rectangle((x[j], y + 0.02), x[j + 1] - x[j], H / 2 - 0.02,
                                   color=cm_img(0.08 + 0.86 * si[j]), lw=0))
            ax.add_patch(Rectangle((x[j], y - H / 2), x[j + 1] - x[j], H / 2 - 0.02,
                                   color=cm_txt(0.08 + 0.86 * st[j]), lw=0))
        if pb:
            ax.plot([pb, pb], [y - H / 2 - 0.07, y + H / 2 + 0.07], color="white",
                    lw=2.6, zorder=4, solid_capstyle="butt")
            ax.plot([pb, pb], [y - H / 2 - 0.07, y + H / 2 + 0.07], color=INK,
                    lw=1.1, ls=(0, (2.6, 1.8)), zorder=5)
            ax.plot(pb, y + H / 2 + 0.13, marker="v", ms=4.5, color=INK, zorder=5)
        # fusion box-whisker
        bw = dict(color=INK, lw=1.6, solid_capstyle="butt", zorder=6)
        ax.add_patch(Rectangle((fq["q25"], y - H / 2 - 0.045), fq["q75"] - fq["q25"],
                               H + 0.09, fill=False, edgecolor=INK, lw=1.6, zorder=6))
        ax.plot([fq["q10"], fq["q25"]], [y, y], **bw)
        ax.plot([fq["q75"], fq["q90"]], [y, y], **bw)
        for w in ("q10", "q90"):
            ax.plot([fq[w], fq[w]], [y - H / 4, y + H / 4], **bw)
        ax.plot(fq["q50"], y, "o", ms=8.5, mfc="white", mec=INK, mew=1.6, zorder=7)
        yticks.append(y)
        ylabels.append(name)
        y -= (H + GAP)

    # group side-labels
    for fam, label in (("native", "NATIVE"), ("encoder", "ENCODER")):
        ys = group_anchor[fam]
        ax.text(-0.225, mean(ys), label, rotation=90, va="center", ha="center",
                fontsize=12, fontweight="bold", color=INK, alpha=0.85,
                transform=ax.transData)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=11.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(y + 0.25, 0.55)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1"], fontsize=10.5)
    for xv in (0.25, 0.5, 0.75):
        ax.axvline(xv, color=INK, lw=0.4, alpha=0.14, zorder=0)
    ax.set_xlabel("relative depth  ℓ / N", fontsize=12)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_linewidth(0.8)

    # compact legend row under the plot
    fig.subplots_adjust(left=0.21, bottom=0.2, right=0.97, top=0.985)
    grad = np.linspace(0, 1, 128)[None, :]
    items = [("grad", cm_img, "image-stream work / layer", 0.13),
             ("grad", cm_txt, "text-stream work / layer", 0.40),
             ("box", None, "fusion depth (whiskers q10–q90,\nbox q25–q75, dot median)", 0.665)]
    for kind, cmap, lab, lx in items:
        if kind == "grad":
            a = fig.add_axes([lx, 0.055, 0.075, 0.018])
            a.imshow(grad, aspect="auto", cmap=cmap, vmin=-0.1, vmax=1.15)
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_lw(0.4)
        else:
            a = fig.add_axes([lx, 0.045, 0.075, 0.038])
            a.set_xlim(0, 1); a.set_ylim(-1, 1)
            a.add_patch(Rectangle((0.3, -0.62), 0.4, 1.24, fill=False, edgecolor=INK, lw=1.3))
            a.plot([0.08, 0.3], [0, 0], color=INK, lw=1.2)
            a.plot([0.7, 0.92], [0, 0], color=INK, lw=1.2)
            a.plot([0.08, 0.08], [-0.4, 0.4], color=INK, lw=1.2)
            a.plot([0.92, 0.92], [-0.4, 0.4], color=INK, lw=1.2)
            a.plot(0.5, 0, "o", ms=6, mfc="white", mec=INK, mew=1.3)
            a.axis("off")
        fig.text(lx + 0.085, 0.064, lab, fontsize=9.5, va="center")

    fig.savefig(OUT / "fig_division_of_labor_pretty.png", dpi=170)
    fig.savefig(OUT / "fig_division_of_labor_pretty.pdf")
    print("wrote fig_division_of_labor_pretty.{png,pdf}")


if __name__ == "__main__":
    main()
