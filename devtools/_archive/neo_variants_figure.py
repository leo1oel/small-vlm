"""NEO version x scale fusion comparison: NEO1.0-2B / 1.0-9B / 1.5-2B / 1.5-9B,
retained(d)/R0 and rho(l) on relative depth, each with its own pre-Buffer
boundary (2B: 12/40=0.30, 9B: 6/42=0.143). Tests whether functional-fusion
relative depth (~0.5) is invariant across version and scale, and where each
model's pre-Buffer sits relative to the fusion zone.

Usage: /envs/neo/bin/python devtools/neo_variants_figure.py <out.png>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fusion_crossmodel_figure import curves, load  # noqa: E402

VARIANTS = [
    ("NEO1.0-2B (40L, pb12)", "results_fusion_full_neo.jsonl", "#2ca02c", 12 / 40),
    ("NEO1.0-9B (42L, pb6)", "results_fusion_full_neo_NEO1_0-9B-SFT.jsonl", "#17becf", 6 / 42),
    ("NEO1.5-2B (40L, pb12)", "results_fusion_full_neo_NEO1_5-2B-SFT.jsonl", "#9467bd", 12 / 40),
    ("NEO1.5-9B (42L, pb6)", "results_fusion_full_neo_NEO1_5-9B-SFT.jsonl", "#e377c2", 6 / 42),
]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = []
    for name, path, color, pb in VARIANTS:
        rows = load(path)
        if rows is None:
            print(f"  (missing {path})")
            continue
        c = curves(rows)
        data.append((name, color, pb, c))
        print(f"{name}: n={c['n']} intact={c['intact']:.3f} swap={c['swap']:.3f} "
              f"R0={c['R0']:+.3f} funcCoM rel={c['com_rel']:.2f} rho@L1={c['rho'][0]:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for name, color, pb, c in data:
        x = [(i + 1) / c["N"] for i in range(c["N"])]
        rn = [c["retained"][i] / c["R0"] if c["R0"] else 0 for i in range(c["N"])]
        ax[0].plot(x, rn, "-o", ms=3, color=color, label=f"{name}  CoM={c['com_rel']:.2f}")
        ax[1].plot(x, c["rho"], "-o", ms=3, color=color, label=name)
        ax[0].axvline(pb, color=color, ls=":", alpha=.4)
        ax[1].axvline(pb, color=color, ls=":", alpha=.4)
    ax[0].axhline(0.5, color="k", lw=.6, ls="--", alpha=.4)
    ax[0].axvspan(0.35, 0.6, color="gray", alpha=.10)
    ax[0].set_title("retained(d)/R0 by NEO version x scale\n(dotted = each model's pre-Buffer boundary)")
    ax[0].set_xlabel("relative blocked depth d / N"); ax[0].set_ylabel("fraction of image signal")
    ax[0].legend(fontsize=8)
    ax[1].set_title("rho(l): representational content-fusion rate")
    ax[1].set_xlabel("relative depth l / N")
    ax[1].set_ylabel(r"$\|h(I)-h(I')\| / \|h(I)-h(\varnothing)\|$")
    ax[1].legend(fontsize=8)
    fig.suptitle("NEO across version (1.0/1.5) and scale (2B/9B), VMCBench dev", y=1.02)
    fig.tight_layout()
    fig.savefig(sys.argv[1], dpi=130, bbox_inches="tight")
    print(f"saved -> {sys.argv[1]}")


if __name__ == "__main__":
    main()
