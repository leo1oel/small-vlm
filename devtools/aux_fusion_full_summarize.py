"""Summarize all-layer unified fusion metrics and emit a base-vs-aux figure.

  cost(d) curve + swap null band   -> functional onset L_f
  nu(l)  curve + swap null band    -> center of mass CoM
  rho(l) curve                     -> representational onset

Usage: python devtools/aux_fusion_full_summarize.py \
    neo_analysis/results_fusion_full_base2000.jsonl \
    neo_analysis/results_fusion_full_aux2000.jsonl \
    [out_figure.png]
"""

import json
import sys
from statistics import mean, pstdev


def load(path):
    return [json.loads(line) for line in open(path) if '"skip"' not in line]


def col(rows, path, default=None):
    """Pull a nested value list-wise; rows that lack it are skipped."""
    out = []
    for r in rows:
        cur = r
        ok = True
        for p in path:
            if isinstance(cur, list):
                cur = cur[p]
            elif p in cur:
                cur = cur[p]
            else:
                ok = False
                break
        if ok:
            out.append(cur)
    return out


def acc_at(rows, getter):
    pairs = [(getter(r), r["gt"]) for r in rows if getter(r) is not None]
    return mean([p["pred"] == g for p, g in pairs]) if pairs else float("nan")


def summarize_arm(rows, N):
    causal = [r for r in rows if "cost" in r]
    n_all, n_c = len(rows), len(causal)
    intact = acc_at(rows, lambda r: r["intact"])
    swap = acc_at(rows, lambda r: r["swap"])

    # cost(d): accuracy and its swap-null band
    cost = [acc_at(causal, lambda r, d=d: r["cost"][d]) for d in range(N)]
    cost_null = [acc_at(causal, lambda r, d=d: r["cost_null"][d]) for d in range(N)]

    # retained(d) = real - null = the usable image signal still alive after
    # blocking text->vision in [0..d).  retained(0) ~ intact - swap.
    retained = [cost[d] - cost_null[d] for d in range(N)]
    R0 = intact - swap  # full usable image signal (d=0)
    # functional onset: half-signal depth (retained drops below R0/2) and
    # 95%-erased depth (retained below 5% of R0).
    L_f50 = next((d + 1 for d in range(N) if retained[d] < 0.5 * R0), None)
    L_f95 = next((d + 1 for d in range(N) if retained[d] < 0.05 * R0), None)

    # marginal functional damage of extending the block by one layer:
    # how much usable signal layer d's cross-modal read was carrying.
    marg = [max(retained[d - 1] - retained[d], 0.0) if d > 0 else max(R0 - retained[0], 0.0)
            for d in range(N)]
    tot = sum(marg) or 1e-9
    CoM = sum((d + 1) * marg[d] for d in range(N)) / tot  # functional fusion center

    # nu(l): single-layer causal KL and its swap null (expected ~0 = redundancy)
    nu = [mean(col(causal, ["nu", l])) for l in range(N)]
    nu_null = [mean(col(causal, ["nu_null", l])) for l in range(N)]
    nu_net = sum(max(nu[l] - nu_null[l], 0.0) for l in range(N))

    # rho(l): content-fusion rate = ||h(I)-h(I')|| / ||h(I)-h(no image)||.
    # High early (cross-modal read dominates the small residual), decays as
    # content-independent image processing accumulates, settles at a floor.
    rho = []
    for l in range(N):
        dsw = col(rows, ["rho", "dswap", l])
        dno = col(rows, ["rho", "dnoimg", l])
        rho.append(mean(dsw) / mean(dno) if dno and mean(dno) > 0 else float("nan"))
    rho1 = rho[0]
    rho_floor = min(rho)
    rho_min_layer = rho.index(rho_floor) + 1
    # representational consolidation depth: rho falls to the midpoint between
    # its layer-1 peak and floor -> where content-fusion stops dominating.
    mid = (rho1 + rho_floor) / 2
    rho_consol = next((l + 1 for l in range(N) if rho[l] <= mid), None)

    return dict(n_all=n_all, n_causal=n_c, intact=intact, swap=swap,
                cost=cost, cost_null=cost_null, retained=retained, R0=R0,
                L_f50=L_f50, L_f95=L_f95, marg=marg, CoM=CoM,
                nu=nu, nu_null=nu_null, nu_net=nu_net,
                rho=rho, rho1=rho1, rho_floor=rho_floor,
                rho_min_layer=rho_min_layer, rho_consol=rho_consol)


def main():
    base_rows, aux_rows = load(sys.argv[1]), load(sys.argv[2])
    N = len(base_rows[0]["rho"]["dswap"])
    B, A = summarize_arm(base_rows, N), summarize_arm(aux_rows, N)

    print(f"N_layers={N}  base(n={B['n_all']}, causal={B['n_causal']})  "
          f"aux(n={A['n_all']}, causal={A['n_causal']})")
    print(f"\nintact acc : base {B['intact']:.3f}  aux {A['intact']:.3f}")
    print(f"swap   acc : base {B['swap']:.3f}  aux {A['swap']:.3f}  (wrong-image floor)")
    print(f"R0 signal  : base {B['R0']:+.3f}  aux {A['R0']:+.3f}  (intact-swap = usable image signal)")
    print("\n== headline metrics (functional fusion lives where blocking erases image signal) ==")
    print(f"L_f50 half-signal depth : base {B['L_f50']}  aux {A['L_f50']}")
    print(f"L_f95 95%-erased depth  : base {B['L_f95']}  aux {A['L_f95']}")
    print(f"functional CoM (layer)  : base {B['CoM']:.2f}  aux {A['CoM']:.2f}   (over marginal cost damage)")
    print(f"single-layer nu net     : base {B['nu_net']:.4f}  aux {A['nu_net']:.4f}   (~0 = redundant, no single-layer bottleneck)")
    print("\n== representational content-fusion rate rho ==")
    print(f"rho @ L1        : base {B['rho1']:.3f}  aux {A['rho1']:.3f}   (image content in text stream from layer 1)")
    print(f"rho floor       : base {B['rho_floor']:.3f}@L{B['rho_min_layer']}  aux {A['rho_floor']:.3f}@L{A['rho_min_layer']}")
    print(f"consolidation L : base {B['rho_consol']}  aux {A['rho_consol']}   (rho falls to peak-floor midpoint)")

    print("\n== per-layer table ==")
    print(f"{'l':>3} | {'cost_b':>7} {'cnull_b':>7} {'ret_b':>6} | {'cost_a':>7} {'cnull_a':>7} {'ret_a':>6} | "
          f"{'nu_b':>7} {'nu_a':>7} | {'rho_b':>6} {'rho_a':>6}")
    for l in range(N):
        print(f"{l+1:>3} | {B['cost'][l]:7.3f} {B['cost_null'][l]:7.3f} {B['retained'][l]:+6.3f} | "
              f"{A['cost'][l]:7.3f} {A['cost_null'][l]:7.3f} {A['retained'][l]:+6.3f} | "
              f"{B['nu'][l]:7.4f} {A['nu'][l]:7.4f} | "
              f"{B['rho'][l]:6.3f} {A['rho'][l]:6.3f}")

    if len(sys.argv) > 3:
        make_figure(B, A, N, sys.argv[3])


def make_figure(B, A, N, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = list(range(1, N + 1))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    cb, ca = "#1f77b4", "#d62728"

    ax[0].plot(x, B["cost"], "-o", ms=3, color=cb, label="base (real img)")
    ax[0].plot(x, A["cost"], "-o", ms=3, color=ca, label="aux (real img)")
    ax[0].plot(x, B["cost_null"], "--", color=cb, alpha=.45, label="base null (swap img)")
    ax[0].plot(x, A["cost_null"], "--", color=ca, alpha=.45, label="aux null (swap img)")
    for arm, c in ((B, cb), (A, ca)):
        if arm["L_f50"]:
            ax[0].axvline(arm["L_f50"], color=c, ls=":", alpha=.6)
    ax[0].set_title("cost(d): cumulative decouplability\n(dotted = L_f50 half-signal depth)")
    ax[0].set_xlabel("blocked depth d  [0..d)"); ax[0].set_ylabel("VMCBench acc")
    ax[0].legend(fontsize=7)

    ax[1].plot(x, B["retained"], "-o", ms=3, color=cb, label="base")
    ax[1].plot(x, A["retained"], "-o", ms=3, color=ca, label="aux")
    ax[1].axhline(0, color="k", lw=.6, alpha=.4)
    ax[1].axhline(B["R0"] / 2, color=cb, ls=":", alpha=.4)
    ax[1].axhline(A["R0"] / 2, color=ca, ls=":", alpha=.4)
    ax[1].set_title("retained(d) = real - null\n(usable image signal surviving the block)")
    ax[1].set_xlabel("blocked depth d  [0..d)"); ax[1].set_ylabel("acc(real) - acc(swap)")
    ax[1].legend(fontsize=7)

    ax[2].plot(x, B["rho"], "-o", ms=3, color=cb, label="base")
    ax[2].plot(x, A["rho"], "-o", ms=3, color=ca, label="aux")
    ax[2].set_title("rho(l): representational fusion rate"); ax[2].set_xlabel("layer l")
    ax[2].set_ylabel(r"$\|h(I)-h(I')\| / \|h(I)-h(\varnothing)\|$"); ax[2].legend(fontsize=7)

    fig.suptitle("Unified fusion metrics — bee-mix base vs aux (k=6, λ=0.5), VMCBench dev-1000", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nsaved figure -> {out}")


if __name__ == "__main__":
    main()
