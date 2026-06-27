"""NEO PT/MT/SFT fusion-location evolution: overlay retained(d)/R0 and rho(l)
across training stages on relative depth. Functional (retained) is reliable
only where the stage actually answers MCQ (PT is a caption model -> its
accuracy-based cost is near-chance; rho is stage-robust).

Usage: /envs/neo/bin/python devtools/neo_stage_figure.py <out.png>
"""

import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1] / "neo_analysis"
STAGES = [
    ("PT", "results_fusion_full_neo_PT.jsonl", "#9467bd"),
    ("MT", "results_fusion_full_neo_MT.jsonl", "#ff7f0e"),
    ("SFT", "results_fusion_full_neo.jsonl", "#2ca02c"),
]
PREBUF_REL = 12 / 40


def load(p):
    f = ROOT / p
    if not f.exists():
        return None
    return [json.loads(l) for l in open(f) if '"skip"' not in l]


def curves(rows):
    causal = [r for r in rows if "cost" in r]
    N = len(rows[0]["rho"]["dswap"])

    def acc(rs, g):
        pr = [(g(r), r["gt"]) for r in rs]
        return mean(p["pred"] == gt for p, gt in pr)

    intact = acc(causal, lambda r: r["intact"])
    swap = acc(causal, lambda r: r["swap"])
    R0 = intact - swap
    retained = [
        acc(causal, lambda r, d=d: r["cost"][d]) - acc(causal, lambda r, d=d: r["cost_null"][d])
        for d in range(N)
    ]
    rho = [
        mean(r["rho"]["dswap"][l] for r in rows) / mean(r["rho"]["dnoimg"][l] for r in rows)
        for l in range(N)
    ]
    marg = [max((R0 if d == 0 else retained[d - 1]) - retained[d], 0.0) for d in range(N)]
    tot = sum(marg) or 1e-9
    com = sum((d + 1) * marg[d] for d in range(N)) / tot
    return dict(
        N=N,
        R0=R0,
        intact=intact,
        swap=swap,
        retained=retained,
        rho=rho,
        com=com,
        com_rel=com / N,
        n=len(rows),
    )


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = []
    for name, path, color in STAGES:
        rows = load(path)
        if rows is None:
            print(f"  (missing {path})")
            continue
        c = curves(rows)
        data.append((name, color, c))
        print(
            f"NEO-{name}: n={c['n']} intact={c['intact']:.3f} swap={c['swap']:.3f} "
            f"R0={c['R0']:+.3f} funcCoM=L{c['com']:.1f} (rel {c['com_rel']:.2f}) "
            f"rho@L1={c['rho'][0]:.3f} rho_floor={min(c['rho']):.3f}"
        )

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for name, color, c in data:
        x = [(i + 1) / c["N"] for i in range(c["N"])]
        rn = [c["retained"][i] / c["R0"] if c["R0"] else 0 for i in range(c["N"])]
        ax[0].plot(
            x,
            rn,
            "-o",
            ms=3,
            color=color,
            label=f"NEO-{name} (R0={c['R0']:+.2f}, CoM rel {c['com_rel']:.2f})",
        )
        ax[1].plot(x, c["rho"], "-o", ms=3, color=color, label=f"NEO-{name}")
    for a in ax:
        a.axvline(PREBUF_REL, color="k", ls=":", alpha=0.5)
        a.axvspan(0.35, 0.6, color="gray", alpha=0.10)
    ax[0].axhline(0.5, color="k", lw=0.6, ls="--", alpha=0.4)
    ax[0].set_title(
        "retained(d)/R0 by stage (dotted = pre-Buffer L12)\n"
        "(PT is a caption model -> accuracy-cost unreliable)"
    )
    ax[0].set_xlabel("relative blocked depth d / N")
    ax[0].set_ylabel("fraction of image signal")
    ax[0].legend(fontsize=8)
    ax[1].set_title("rho(l): representational content-fusion rate by stage")
    ax[1].set_xlabel("relative depth l / N")
    ax[1].set_ylabel(r"$\|h(I)-h(I')\| / \|h(I)-h(\varnothing)\|$")
    ax[1].legend(fontsize=8)
    fig.suptitle(
        "NEO-2B fusion location across training stages (PT->MT->SFT), VMCBench dev", y=1.02
    )
    fig.tight_layout()
    fig.savefig(sys.argv[1], dpi=130, bbox_inches="tight")
    print(f"saved -> {sys.argv[1]}")


if __name__ == "__main__":
    main()
