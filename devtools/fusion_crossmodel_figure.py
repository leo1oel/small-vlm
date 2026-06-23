"""Cross-model fusion figure: overlay retained(d) and rho(l) for bee-mix /
NEO-2B / Gemma-4-12B on RELATIVE depth (layer/N), so the functional-fusion
location is comparable across 28/40/48-layer stacks.

Usage: python devtools/fusion_crossmodel_figure.py <out.png>
Reads the three results_fusion_full_*.jsonl from neo_analysis/.
Run with the neo venv python (has matplotlib): /envs/neo/bin/python.
"""

import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1] / "neo_analysis"
MODELS = [
    ("bee-mix 2B (28L)", "results_fusion_full_base5000.jsonl", "#1f77b4", None),
    ("NEO1.0-2B (40L)", "results_fusion_full_neo.jsonl", "#2ca02c", 12 / 40),
    ("NEO1.0-9B (42L)", "results_fusion_full_neo_NEO1_0-9B-SFT.jsonl", "#17becf", 6 / 42),
    ("Gemma-4-12B (48L)", "results_fusion_full_gemma4.jsonl", "#d62728", None),
]


def load(path):
    p = ROOT / path
    if not p.exists():
        return None
    return [json.loads(l) for l in open(p) if '"skip"' not in l]


def curves(rows):
    causal = [r for r in rows if "cost" in r]
    N = len(rows[0]["rho"]["dswap"])

    def acc(rs, g):
        pr = [(g(r), r["gt"]) for r in rs]
        return mean(p["pred"] == gt for p, gt in pr)

    intact = acc(causal, lambda r: r["intact"])
    swap = acc(causal, lambda r: r["swap"])
    retained = [acc(causal, lambda r, d=d: r["cost"][d]) -
                acc(causal, lambda r, d=d: r["cost_null"][d]) for d in range(N)]
    R0 = intact - swap
    rho = []
    for l in range(N):
        dsw = mean(r["rho"]["dswap"][l] for r in rows)
        dno = mean(r["rho"]["dnoimg"][l] for r in rows)
        rho.append(dsw / dno if dno > 0 else float("nan"))
    # functional CoM over marginal retained drop
    marg = [max((R0 if d == 0 else retained[d - 1]) - retained[d], 0.0) for d in range(N)]
    tot = sum(marg) or 1e-9
    com = sum((d + 1) * marg[d] for d in range(N)) / tot
    return dict(N=N, R0=R0, intact=intact, swap=swap, retained=retained, rho=rho,
                com=com, com_rel=com / N, n=len(rows), nc=len(causal))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = []
    for name, path, color, prebuf in MODELS:
        rows = load(path)
        if rows is None:
            print(f"  (missing {path}, skipping {name})")
            continue
        c = curves(rows)
        data.append((name, color, prebuf, c))
        print(f"{name}: n={c['n']} causal={c['nc']} intact={c['intact']:.3f} "
              f"swap={c['swap']:.3f} R0={c['R0']:+.3f} funcCoM=L{c['com']:.1f} "
              f"(rel {c['com_rel']:.2f}) rho@L1={c['rho'][0]:.3f} rho_floor={min(c['rho']):.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for name, color, prebuf, c in data:
        x = [(i + 1) / c["N"] for i in range(c["N"])]
        # retained normalized to its own R0 -> fraction of usable image signal alive
        rn = [c["retained"][i] / c["R0"] if c["R0"] else 0 for i in range(c["N"])]
        ax[0].plot(x, rn, "-o", ms=3, color=color, label=f"{name} (CoM rel {c['com_rel']:.2f})")
        ax[1].plot(x, c["rho"], "-o", ms=3, color=color, label=name)
        if prebuf:
            ax[0].axvline(prebuf, color=color, ls=":", alpha=.4)
            ax[1].axvline(prebuf, color=color, ls=":", alpha=.4)
    ax[0].axhline(0.5, color="k", lw=.6, ls="--", alpha=.4)
    ax[0].axhspan(0.35, 0.6, color="gray", alpha=.10)
    ax[0].set_title("retained(d)/R0: usable image signal surviving block [0..d)\n"
                    "(gray = relative depth 0.35-0.60)")
    ax[0].set_xlabel("relative blocked depth  d / N"); ax[0].set_ylabel("fraction of image signal")
    ax[0].legend(fontsize=8)
    ax[1].axhspan(0.35, 0.6, color="gray", alpha=.0)
    ax[1].set_title("rho(l): representational content-fusion rate")
    ax[1].set_xlabel("relative depth  l / N")
    ax[1].set_ylabel(r"$\|h(I)-h(I')\| / \|h(I)-h(\varnothing)\|$")
    ax[1].legend(fontsize=8)
    fig.suptitle("Where does functional fusion live? — cross-model, VMCBench dev (dotted = NEO pre-Buffer boundary)",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(sys.argv[1], dpi=130, bbox_inches="tight")
    print(f"saved -> {sys.argv[1]}")


if __name__ == "__main__":
    main()
