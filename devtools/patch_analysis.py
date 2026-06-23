"""Analyse activation-patch results and triangulate against the attention-knockout fusion depth.

For each model's results_patch_*.jsonl:
  R0          = intact_acc - swap_acc  (over causal rows, same as fusion_window)
  denoise:  retained_dn(d)  = (acc(denoise[d]) - swap_acc)/R0          # rises 0 -> 1
            -> a SUFFICIENCY/onset distribution (marginal RISE), triangulates the SUFFIX onset.
  meanabl:  retained_mab(d) = (acc(meanabl[d]) - acc(meanabl_null[d]))/R0   # falls 1 -> 0
            -> a NECESSITY distribution (marginal FALL), triangulates the PREFIX knockout phi.
Quantiles q10/q25/q50/q75/q90 are computed in relative depth d/N with the SAME CDF method as
devtools/fig_two_conclusions.py::fusion_quants, so the numbers are directly comparable.

Optionally pass a knockout jsonl (with cost/cost_null and/or suf/suf_null) to print phi/onset
computed from the same fusion_quants method on the same model for a side-by-side.

Usage: python devtools/patch_analysis.py <results_patch.jsonl> <tag> [knockout.jsonl]
"""
import json
import sys
from statistics import mean

# report reference phi (prefix-knockout median, from neo_report) for a sanity cross-check
REF_PHI = {"llava": 0.33, "qwen": 0.54, "neo": 0.54, "sail": 0.59, "gemma": 0.48}


def load(path):
    return [json.loads(l) for l in open(path) if l.strip() and '"skip"' not in l]


def quants_from_retained(ret, rising):
    """ret: list length N+1 (d=0..N). Returns q10..q90 in relative depth d/N."""
    N = len(ret) - 1
    if rising:
        marg = [max(ret[d] - ret[d - 1], 0) for d in range(1, N + 1)]
    else:
        marg = [max(ret[d - 1] - ret[d], 0) for d in range(1, N + 1)]
    tot = sum(marg) or 1.0
    cdf, s = [], 0.0
    for m in marg:
        s += m
        cdf.append(s / tot)

    def q(p):
        return next((d / N for d in range(1, N + 1) if cdf[d - 1] >= p), 1.0)
    return {k: q(v) for k, v in (("q10", .1), ("q25", .25), ("q50", .5), ("q75", .75), ("q90", .9))}


def patch_quants(rows):
    cz = [r for r in rows if "denoise" in r or "meanabl" in r or "sufmeanabl" in r]
    out = {"n_causal": len(cz)}
    if not cz:
        return out
    iacc = mean(r["intact"]["pred"] == r["gt"] for r in cz)
    sacc = mean(r["swap"]["pred"] == r["gt"] for r in cz)
    out["intact_acc"], out["swap_acc"], out["R0"] = round(iacc, 3), round(sacc, 3), round(iacc - sacc, 3)
    if iacc - sacc < 0.05:
        out["warn"] = "R0<0.05 — curves are noise"
    # each metric on the subset of causal rows that actually carry it (NEO/SAIL skip denoise)
    dz = [r for r in cz if "denoise" in r]
    if dz:
        R0 = mean(r["intact"]["pred"] == r["gt"] for r in dz) - mean(r["swap"]["pred"] == r["gt"] for r in dz)
        sd = mean(r["swap"]["pred"] == r["gt"] for r in dz)
        Nd = len(dz[0]["depths"]) - 1
        ret = [(mean(r["denoise"][d]["pred"] == r["gt"] for r in dz) - sd) / R0 for d in range(Nd + 1)]
        out["denoise"] = quants_from_retained(ret, rising=True)
        out["n_denoise"] = len(dz)
    mz = [r for r in cz if "meanabl" in r]
    if mz:
        R0 = mean(r["intact"]["pred"] == r["gt"] for r in mz) - mean(r["swap"]["pred"] == r["gt"] for r in mz)
        Nd = len(mz[0]["depths"]) - 1
        ret = [(mean(r["meanabl"][d]["pred"] == r["gt"] for r in mz)
                - mean(r["meanabl_null"][d]["pred"] == r["gt"] for r in mz)) / R0 for d in range(Nd + 1)]
        out["meanabl"] = quants_from_retained(ret, rising=False)
        out["n_meanabl"] = len(mz)
    sm = [r for r in cz if "sufmeanabl" in r]
    if sm:
        R0 = mean(r["intact"]["pred"] == r["gt"] for r in sm) - mean(r["swap"]["pred"] == r["gt"] for r in sm)
        Nd = len(sm[0]["depths"]) - 1
        # suffix-flatten: retained rises 0->1; marginal RISE = read-onset depth (the primary metric)
        ret = [(mean(r["sufmeanabl"][d]["pred"] == r["gt"] for r in sm)
                - mean(r["sufmeanabl_null"][d]["pred"] == r["gt"] for r in sm)) / R0 for d in range(Nd + 1)]
        out["sufmeanabl"] = quants_from_retained(ret, rising=True)
        out["n_sufmeanabl"] = len(sm)
    return out


def knockout_phi(rows):
    """fusion_quants (prefix phi + suffix onset) from a knockout jsonl, same method as the figures."""
    res = {}
    causal = [r for r in rows if "cost" in r]
    if causal:
        N = len(causal[0]["cost"])

        def acc(g):
            return mean(g(r)["pred"] == r["gt"] for r in causal)
        R0 = acc(lambda r: r["intact"]) - acc(lambda r: r["swap"])
        rn = [(acc(lambda r, d=d: r["cost"][d]) - acc(lambda r, d=d: r["cost_null"][d])) / R0
              for d in range(N)]
        res["phi"] = quants_from_retained([1.0] + rn, rising=False)
    sf = [r for r in rows if "suf" in r]
    if sf:
        N = len(sf[0]["suf"])

        def acc(g):
            return mean(g(r)["pred"] == r["gt"] for r in sf)
        R0 = acc(lambda r: r["intact"]) - acc(lambda r: r["swap"])
        rs = [(acc(lambda r, d=d: r["suf"][d]) - acc(lambda r, d=d: r["suf_null"][d])) / R0
              for d in range(N)]
        res["onset"] = quants_from_retained(rs + [1.0], rising=True)
    return res


def fmt(q):
    return "q50=%.2f [%.2f,%.2f]" % (q["q50"], q["q25"], q["q75"]) if q else "-"


def main():
    path, tag = sys.argv[1], sys.argv[2]
    pq = patch_quants(load(path))
    print(f"== {tag} ==  n_causal={pq.get('n_causal')}  intact={pq.get('intact_acc')} "
          f"swap={pq.get('swap_acc')} R0={pq.get('R0')} {pq.get('warn','')}")
    print(f"  ** sufmeanabl (PRIMARY: suffix-flatten read-onset) [n={pq.get('n_sufmeanabl','-')}]: {fmt(pq.get('sufmeanabl'))}")
    print(f"  meanabl  (prefix-flatten; NON-discriminative)     [n={pq.get('n_meanabl','-')}]: {fmt(pq.get('meanabl'))}")
    print(f"  denoise  (prefix-inject; NON-discriminative)      [n={pq.get('n_denoise','-')}]: {fmt(pq.get('denoise'))}")
    print(f"  [report reference prefix-phi q50 for {tag}: {REF_PHI.get(tag,'?')}]")
    if len(sys.argv) > 3:
        ko = knockout_phi(load(sys.argv[3]))
        print(f"  knockout prefix-phi: {fmt(ko.get('phi'))}   suffix-onset: {fmt(ko.get('onset'))}")


if __name__ == "__main__":
    main()
