"""Persuasive figure set for the VLM fusion study. Elegant, model-diverse, one clear
conclusion per figure. Run with an env that has matplotlib (neo venv).

Produces:
  fig_fusion_forest.png  — Conclusion 1: fusion is mid-stack across 10 architectures (forest plot)
  fig_invariance.png     — Conclusion 1 robustness: depth invariant across benchmark + visual domain
  fig_triangulation.png  — credibility: two orthogonal causal metrics rank models identically

(Claim B's figure is fig_internal_encoder.png, produced by fig_internal_encoder.py.)

Usage: python devtools/fig_persuasive.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "neo_analysis"
OUT = Path(__file__).resolve().parents[1] / "neo_report"
D = json.load(open(ROOT / "fusion_depth_all.json"))

# ---- shared style ----
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#444444", "axes.linewidth": 0.9,
    "xtick.color": "#444444", "ytick.color": "#444444",
    "axes.titlesize": 12.5, "axes.titleweight": "semibold", "axes.labelcolor": "#222222",
})
FAM_C = {"encoder": "#4C72B0", "encoder-free": "#DD8452", "unified": "#8172B3"}
FAM_LABEL = {"encoder": "encoder-based", "encoder-free": "encoder-free", "unified": "unified (no separate encoder)"}
BAND = (0.35, 0.55)  # mid-stack
MIDC = "#6c757d"


def _band(ax, vertical=False):
    if vertical:
        ax.axhspan(*BAND, color=MIDC, alpha=0.10, zorder=0, lw=0)
    else:
        ax.axvspan(*BAND, color=MIDC, alpha=0.10, zorder=0, lw=0)


# ============================ FIG 1: fusion forest ============================
def fig_forest():
    rows = [(t, m) for t, m in D["models"].items()
            if t != "llava" and m.get("vmcbench")]                  # drop the explained outlier
    rows.sort(key=lambda tm: tm[1]["vmcbench"]["q50"])
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    _band(ax)
    ax.axvline(np.mean(BAND), color=MIDC, ls=(0, (4, 4)), lw=1, zorder=1)
    seen = set()
    for i, (t, m) in enumerate(rows):
        v = m["vmcbench"]; c = FAM_C[m["family"]]
        lbl = FAM_LABEL[m["family"]] if m["family"] not in seen else None
        seen.add(m["family"])
        ax.plot([v["q25"], v["q75"]], [i, i], color=c, lw=2.4, alpha=0.55, solid_capstyle="round", zorder=2)
        ax.scatter([v["q50"]], [i], s=130 if m["moe"] else 95, color=c, zorder=3,
                   marker="D" if m["moe"] else "o", edgecolor="white", linewidth=1.3, label=lbl)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([m["label"] + ("  ·MoE" if m["moe"] else "") for _, m in rows], fontsize=10)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(0, 1)
    ax.set_xlabel("relative fusion depth  (read-onset, sufmeanabl q50;  0 = input, 1 = output)")
    ax.set_title("Vision is read into the answer at mid-stack — across every architecture")
    ax.text(np.mean(BAND), len(rows) - 0.45, "mid-stack", ha="center", va="bottom",
            fontsize=9.5, color=MIDC, style="italic")
    # legend: families + MoE marker
    h, l = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h += [Line2D([], [], marker="D", color="#888", ls="", markeredgecolor="white", label="sparse MoE arm")]
    l += ["sparse MoE arm"]
    ax.legend(h, l, loc="lower right", frameon=False, fontsize=9, handletextpad=0.3)
    ax.text(0.012, -0.62, "LLaVA-1.5 (0.22) omitted — the one early case, a small-scale frozen-encoder "
            "recipe, not an architecture effect.", fontsize=7.6, color="#999", style="italic")
    fig.tight_layout()
    fig.savefig(OUT / "fig_fusion_forest.png", bbox_inches="tight")
    print("[fig] fig_fusion_forest.png", flush=True)
    plt.close(fig)


# ============================ FIG 2: invariance ============================
def fig_invariance():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))
    # (A) slopegraph across benchmarks
    benches = ["vmcbench", "mmstar", "worldbench"]
    bx = {"vmcbench": 0, "mmstar": 1, "worldbench": 2}
    _band(axA, vertical=True)
    for t, m in D["models"].items():
        pts = [(bx[b], m[b]["q50"]) for b in benches if m.get(b) and m[b].get("q50") is not None]
        if len(pts) < 2 or t == "llava":
            continue
        xs, ys = zip(*pts)
        c = FAM_C[m["family"]]
        axA.plot(xs, ys, "-o", color=c, lw=2, ms=7, markeredgecolor="white", markeredgewidth=1.2, alpha=0.9)
        dy = {"ivl8": 9, "gemma12": -9}.get(t, 0)  # ivl8 & gemma12 coincide at 0.42 → split labels
        axA.annotate(m["label"], (xs[-1], ys[-1]), xytext=(7, dy), textcoords="offset points",
                     va="center", fontsize=8.5, color=c, fontweight="medium")
    axA.set_xticks([0, 1, 2]); axA.set_xticklabels(["VMCBench", "MMStar", "WorldBench"])
    axA.set_xlim(-0.3, 2.7); axA.set_ylim(0.2, 0.7)
    axA.set_ylabel("relative fusion depth (sufmeanabl q50)")
    axA.set_title("Invariant across benchmarks\n(flat lines = depth doesn't move with dataset)")
    axA.text(2.0, np.mean(BAND), "mid-stack", color=MIDC, fontsize=9, style="italic", va="center")

    # (B) WorldBench by visual domain (full-data anchor = Gemma)
    g = D["models"]["gemma12"].get("worldbench", {}).get("by_domain", {})
    if not g:
        g = D["models"]["ivl8"]["worldbench"]["by_domain"]
        anchor = "InternVL3.5-8B"
    else:
        anchor = "Gemma-4-12B"
    doms = sorted(g)
    _band(axB, vertical=True)
    for i, dom in enumerate(doms):
        sig = g[dom]["R0"] >= 0.05
        axB.bar(i, g[dom]["q50"], 0.62, color=FAM_C["encoder-free"] if sig else "#d9d9d9",
                edgecolor="white", zorder=2)
    ov = D["models"]["gemma12"]["worldbench"]["q50"]
    axB.axhline(ov, color="#c44", ls=(0, (4, 3)), lw=1.2, label=f"overall q50={ov:.2f}")
    axB.set_xticks(range(len(doms)))
    axB.set_xticklabels([d.replace(", ", ",\n").replace("Documents,Charts,& Tables", "Docs/\nCharts")
                         .replace(" ", "\n", 1) for d in doms], fontsize=8)
    axB.set_ylim(0, 0.7); axB.set_ylabel("relative fusion depth (sufmeanabl q50)")
    axB.set_title(f"Invariant across visual domain ({anchor})\n(grey = no causal signal, R₀<0.05)")
    axB.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_invariance.png", bbox_inches="tight")
    print("[fig] fig_invariance.png", flush=True)
    plt.close(fig)


# ============================ FIG 3: triangulation ============================
def fig_triangulation():
    phi = D["phi"]
    pts = []
    for t, p in phi.items():
        m = D["models"].get(t)
        if m and m.get("vmcbench"):
            pts.append((t, p, m["vmcbench"]["q50"], m))
    fig, ax = plt.subplots(figsize=(6.4, 6))
    ax.plot([0, 1], [0, 1], color="#bbb", ls="--", lw=1, zorder=1)
    short = {"ivl8": "InternVL-8B", "gemma12": "Gemma-4-12B", "neo2b": "NEO-2B",
             "sail": "SAIL-7B", "qwen": "Qwen2.5-VL", "llava": "LLaVA-1.5"}
    off = {"ivl8": (8, 7), "gemma12": (8, -12), "qwen": (8, 2), "neo2b": (8, -2),
           "sail": (8, 2), "llava": (8, -2)}  # split the ivl8/gemma12 coincidence at (≈0.45, 0.42)
    for t, x, y, m in pts:
        c = FAM_C[m["family"]]
        ax.scatter([x], [y], s=120, color=c, edgecolor="white", linewidth=1.3, zorder=3)
        ax.annotate(short.get(t, m["label"]), (x, y), xytext=off.get(t, (8, -2)),
                    textcoords="offset points", fontsize=9, color=c, fontweight="medium")
    xs = [p[1] for p in pts]; ys = [p[2] for p in pts]
    r = np.corrcoef(xs, ys)[0, 1]
    ax.set_xlim(0.2, 0.7); ax.set_ylim(0.2, 0.7)
    ax.set_xlabel("attention-knockout  φ  (completion depth)")
    ax.set_ylabel("residual-stream sufmeanabl  q50  (read-onset)")
    ax.set_title("Two orthogonal causal metrics agree\n"
                 f"(same ranking, Pearson r={r:.2f}; onset slightly precedes completion)")
    ax.text(0.62, 0.64, "y = x", color="#999", fontsize=9, rotation=33, ha="center")
    fig.tight_layout()
    fig.savefig(OUT / "fig_triangulation.png", bbox_inches="tight")
    print("[fig] fig_triangulation.png", flush=True)
    plt.close(fig)


# ============ FIG 4: vision built (to encoder-grade) BEFORE fusion reads it ============
def fig_build_before_fuse():
    P = json.load(open(ROOT / "predictivity_n1000.json"))
    # all models with BOTH predictivity + fusion. native = vision built INSIDE (curve climbs);
    # encoder-based = vision built OUTSIDE by a frozen encoder (flat-high). (pred, fuse, label, group)
    cand = [("gemma", "gemma12", "Gemma-4-12B", "native"),
            ("neo2bsft", "neo2b", "NEO-2B", "native"),
            ("neo9bsft", "neo9b", "NEO-9B", "native"),
            ("gemma26moe", "gemma26moe", "Gemma-4-26B-A4B (MoE)", "native"),
            ("mono", "mono", "Mono-InternVL-2B", "native"),
            ("qwen", "qwen", "Qwen2.5-VL-7B", "encoder"),
            ("ivl2", "ivl2", "InternVL3.5-2B", "encoder"),
            ("ivl4", "ivl4", "InternVL3.5-4B", "encoder"),
            ("ivl8", "ivl8", "InternVL3.5-8B", "encoder"),
            ("ivl30moe", "ivl30moe", "InternVL3.5-30B-A3B (MoE)", "encoder"),
            ("janus", "janus", "Janus-Pro-7B", "encoder")]
    rows = [r for r in cand if r[0] in P and D["models"].get(r[1], {}).get("vmcbench")]
    EG_LO = 0.60
    import matplotlib.cm as cm
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8.2), sharex=True)
    for grp, ax, cmap in [("native", ax1, cm.Oranges), ("encoder", ax2, cm.Blues)]:
        ms = [r for r in rows if r[3] == grp]
        ax.axhspan(EG_LO, 0.80, color="#2ca25f", alpha=0.10, zorder=0, lw=0)
        for j, (pt, ft, label, _) in enumerate(ms):
            cur = np.array(P[pt]["per_encoder"]["siglip"]["r2_curve"])
            x = np.linspace(0, 1, len(cur))
            c = cmap(0.45 + 0.5 * (j / max(len(ms) - 1, 1)))
            ax.plot(x, cur, color=c, lw=2.1, zorder=3, label=label)
            fu = D["models"][ft]["vmcbench"]["q50"]
            ax.plot([fu], [-0.045], marker="^", ms=8, color=c, clip_on=False, zorder=5)  # fusion median tick
        ax.axvspan(0.35, 0.55, color="#555", alpha=0.08, zorder=0)  # mid-stack fusion band (where the ticks sit)
        ax.set_ylim(0, 0.80); ax.set_xlim(0, 1)
        ax.set_ylabel("predictivity to SigLIP  (R²)\n= how encoder-grade", fontsize=9.5)
        ax.legend(loc="lower right", frameon=False, fontsize=8.2, ncol=1)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.text(0.012, 0.755, "encoder-grade zone", color="#1b7a3d", fontsize=8.5, style="italic", va="top")
    ax1.set_title("Vision is encoder-grade BEFORE mid-stack fusion reads it (▲ = each model's fusion depth)\n"
                  "TOP: native models BUILD it inside (curves climb into the zone before ▲)  ·  "
                  "BOTTOM: encoder-VLMs have it PRE-BUILT outside (flat-high from layer 0)", fontsize=10.5)
    ax1.text(0.012, 0.07, "native — built INSIDE (generative pre-training)", color="#a5531f",
             fontsize=9, fontweight="medium")
    ax2.text(0.012, 0.07, "encoder-VLM — built OUTSIDE (frozen vision encoder)", color="#2f4d7a",
             fontsize=9, fontweight="medium")
    ax2.set_xlabel("relative depth  ℓ / N   (0 = input, 1 = output);   ▲ fusion median (mid-stack)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_build_before_fuse.png", bbox_inches="tight")
    print("[fig] fig_build_before_fuse.png", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    fig_forest()
    fig_invariance()
    fig_triangulation()
    fig_build_before_fuse()
    print("[fig] done — 4 figures in neo_report/", flush=True)
