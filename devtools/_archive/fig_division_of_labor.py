"""THE summary figure: per-model 'division of labor' map over relative depth.

For every model, two heat strips over relative depth x = l/N:
  top strip    = image-stream work  u_img(l) (normalized to its own L1..N mean)
  bottom strip = text-stream work   u_txt(l) (same normalization)
overlaid with the FUSION interval measured causally:
  [ = prefix-knockout onset80 (where blocking starts to bite = necessary fusion
      begins), dot = FDI (depth center of mass), ] = prefix c20 (fusion done).
NEO rows also mark the pre-Buffer boundary (white dashed).

Reads maturation jsonl + fusion (prefix) jsonl per model. Natives >=5, encoder
>=3 per the universality requirement.

Run with neo venv python. Usage: python devtools/fig_division_of_labor.py
"""

import json
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "neo_analysis"
OUT = Path(__file__).resolve().parents[1] / "neo_report"

# name -> (family, mat_file, fusion_file(prefix cost), prebuf_rel or None)
MODELS = [
    (
        "NEO1.0-2B-SFT",
        "native",
        "results_mat_neo_2B-SFT.jsonl",
        "results_fusion_full_neo_strat.jsonl",
        12 / 40,
    ),
    (
        "NEO1.0-2B-MT",
        "native",
        "results_mat_neo_2B-MT.jsonl",
        "results_fusion_full_neo_MT.jsonl",
        12 / 40,
    ),
    (
        "NEO1.0-9B-SFT",
        "native",
        "results_mat_neo_9B-SFT.jsonl",
        "results_fusion_full_neo_NEO1_0-9B-SFT.jsonl",
        6 / 42,
    ),
    (
        "NEO1.0-9B-MT",
        "native",
        "results_mat_neo_9B-MT.jsonl",
        "results_fusion_full_neo_NEO1_0-9B-MT.jsonl",
        6 / 42,
    ),
    (
        "NEO1.5-2B-SFT",
        "native",
        "results_mat_neo_15-2B-SFT.jsonl",
        "results_fusion_full_neo_NEO1_5-2B-SFT.jsonl",
        12 / 40,
    ),
    (
        "NEO1.5-9B-SFT",
        "native",
        "results_mat_neo_15-9B-SFT.jsonl",
        "results_fusion_full_neo_NEO1_5-9B-SFT.jsonl",
        6 / 42,
    ),
    (
        "Gemma-4-12B",
        "native",
        "results_mat_gemma.jsonl",
        "results_fusion_full_gemma4_strat.jsonl",
        None,
    ),
    ("SAIL-7B", "native", "results_mat_sail.jsonl", "results_fusion_full_sail.jsonl", None),
    ("LLaVA-1.5-7B", "encoder", "results_mat_llava.jsonl", "results_fusion_full_llava.jsonl", None),
    (
        "LLaVA-NeXT-7B",
        "encoder",
        "results_mat_llavanext.jsonl",
        "results_win_llavanext.jsonl",
        None,
    ),
    ("OneVision-7B", "encoder", "results_mat_onevision.jsonl", "results_win_onevision.jsonl", None),
    (
        "Qwen2.5-VL-7B",
        "encoder",
        "results_mat_qwen.jsonl",
        "results_fusion_full_qwenvl.jsonl",
        None,
    ),
]


def load(path):
    p = ROOT / path
    if not p.exists() or p.stat().st_size == 0:
        return None
    return [json.loads(l) for l in open(p) if l.strip() and '"skip"' not in l]


def mat_curves(rows):
    rows = [r for r in rows if "u_img" in r]
    if not rows:
        return None
    N = max(len(r["u_img"]) for r in rows)
    rows = [r for r in rows if len(r["u_img"]) == N]
    ui = [mean(r["u_img"][l] for r in rows) for l in range(N)]
    ut = [mean(r["u_txt"][l] for r in rows) for l in range(N)]
    return ui, ut, N


def fusion_marks(rows):
    """Quantiles of the fusion-depth DISTRIBUTION: marg(d) = clipped drop of
    retained(d)/R0 = how much usable image signal becomes unrecoverable at
    depth d. Box = [q25,q75], median = q50, whiskers = [q10,q90] (the latter
    equal 'curve falls below 90% / 10%' read off the monotone CDF)."""
    causal = [r for r in rows if "cost" in r]
    if len(causal) < 20:
        return None
    N = len(causal[0]["cost"])

    def acc(g):
        return mean(g(r)["pred"] == r["gt"] for r in causal)

    R0 = acc(lambda r: r["intact"]) - acc(lambda r: r["swap"])
    if R0 <= 0.02:
        return None
    rn = [
        (acc(lambda r, d=d: r["cost"][d]) - acc(lambda r, d=d: r["cost_null"][d])) / R0
        for d in range(N)
    ]
    marg = [max((1 if d == 0 else rn[d - 1]) - rn[d], 0) for d in range(N)]
    tot = sum(marg)
    if tot <= 0:
        return None
    cdf, s = [], 0.0
    for m in marg:
        s += m
        cdf.append(s / tot)

    def q(p):
        return next(((i + 1) / N for i in range(N) if cdf[i] >= p), 1.0)

    return dict(
        q10=q(0.10), q25=q(0.25), q50=q(0.50), q75=q(0.75), q90=q(0.90), R0=R0, nc=len(causal)
    )


def strip(u, N, clip=2.5):
    # normalize to own mean over L1..N (skip the universal L0 spike), clip for color
    base = mean(u[1:]) or 1e-9
    v = np.array([min(x / base, clip) for x in u])
    v[0] = min(v[0], clip)
    return v


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    rows = []
    for name, fam, mf, ff, pb in MODELS:
        m = load(mf)
        f = load(ff)
        if not m:
            print(f"  skip {name} (no maturation data yet)")
            continue
        mc = mat_curves(m)
        fm = fusion_marks(f) if f else None
        if not mc:
            continue
        rows.append((name, fam, mc, fm, pb))
        if fm:
            print(
                f"{name:16} N={mc[2]:3} fusion q10/25/50/75/90 = "
                f"{fm['q10']:.2f}/{fm['q25']:.2f}/{fm['q50']:.2f}/{fm['q75']:.2f}/{fm['q90']:.2f} "
                f"R0={fm['R0']:+.2f} nc={fm['nc']}"
            )

    H = 0.8
    fig, ax = plt.subplots(figsize=(11, 0.62 * len(rows) + 2.4))
    yticks, ylabels = [], []
    y = 0
    group_sep = []
    last_fam = None
    for name, fam, (ui, ut, N), fm, pb in rows:
        if last_fam and fam != last_fam:
            group_sep.append(y + 0.275)  # midway between previous strip bottom and next top
            y -= 0.45
        last_fam = fam
        x = np.linspace(0, 1, N + 1)
        si = strip(ui, N)
        st = strip(ut, N)
        for j in range(N):
            ax.add_patch(
                Rectangle(
                    (x[j], y),
                    x[j + 1] - x[j],
                    H / 2,
                    color=plt.cm.Reds(min(si[j] / 2.5, 1.0)),
                    lw=0,
                )
            )
            ax.add_patch(
                Rectangle(
                    (x[j], y - H / 2),
                    x[j + 1] - x[j],
                    H / 2,
                    color=plt.cm.Purples(min(st[j] / 2.5, 1.0)),
                    lw=0,
                )
            )
        if fm:
            # box-and-whisker of the fusion-depth distribution
            for e in ("q25", "q75"):
                ax.plot([fm[e], fm[e]], [y - H / 2, y + H / 2], color="k", lw=2)
            ax.plot([fm["q25"], fm["q75"]], [y + H / 2, y + H / 2], color="k", lw=1.2)
            ax.plot([fm["q25"], fm["q75"]], [y - H / 2, y - H / 2], color="k", lw=1.2)
            ax.plot([fm["q10"], fm["q25"]], [y, y], color="k", lw=1)
            ax.plot([fm["q75"], fm["q90"]], [y, y], color="k", lw=1)
            for w in ("q10", "q90"):
                ax.plot([fm[w], fm[w]], [y - H / 4, y + H / 4], color="k", lw=1)
            ax.plot(fm["q50"], y, "o", color="k", ms=8, mfc="yellow", zorder=5)
        if pb:
            ax.plot([pb, pb], [y - H / 2, y + H / 2], color="w", lw=1.6, ls="--", zorder=4)
        yticks.append(y)
        ylabels.append(f"{name}")
        y -= 1
    for gs in group_sep:
        ax.axhline(gs, color="k", lw=0.8)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(y + 0.3, 0.8)
    ax.set_xlabel("relative depth l/N")
    n_nat = sum(1 for r in rows if r[1] == "native")
    n_enc = len(rows) - n_nat
    ax.set_title(
        f"Division of labor over depth — {n_nat} native + {n_enc} encoder VLMs\n"
        "each row: TOP strip = image-stream work u_img (red), BOTTOM = text-stream work u_txt (purple), "
        "each normalized to its own per-layer mean\n"
        "black box-and-whisker = FUSION-depth distribution (where blocking text→image attention "
        "destroys usable image signal):\n"
        "box = middle 50% [q25,q75], yellow dot = median, whiskers = [q10,q90]; "
        "white dashed = NEO pre-Buffer boundary",
        fontsize=9,
    )
    # annotate groups
    ax.text(
        1.015,
        mean([yticks[i] for i in range(n_nat)]),
        "NATIVE\n(no encoder)",
        rotation=90,
        va="center",
        fontsize=9,
        fontweight="bold",
    )
    ax.text(
        1.015,
        mean([yticks[i] for i in range(n_nat, len(rows))]),
        "ENCODER",
        rotation=90,
        va="center",
        fontsize=9,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_division_of_labor.png", dpi=140, bbox_inches="tight")
    print("wrote fig_division_of_labor.png")


if __name__ == "__main__":
    main()
