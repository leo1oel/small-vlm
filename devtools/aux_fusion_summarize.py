"""Summarize aux_fusion_probe results: base-2000 vs aux-2000.

Usage: python devtools/aux_fusion_summarize.py \
           neo_analysis/results_fusion_probe_base2000.jsonl \
           neo_analysis/results_fusion_probe_aux2000.jsonl
"""

import json
import sys
from math import comb

DEPTHS = [2, 4, 6, 8, 12, 16, 28]
EXITS = [2, 4, 6, 8, 12, 16, 20, 24]


def load(path):
    recs = [json.loads(line) for line in open(path)]
    return {r["i"]: r for r in recs if "skip" not in r}


def acc(rows, get):
    pairs = [(get(r), r) for r in rows if get(r) is not None]
    return sum(s["pred"] == r["gt"] for s, r in pairs) / max(len(pairs), 1)


def mean_nll(rows, get):
    vals = [get(r)["nll"] for r in rows if get(r) is not None]
    return sum(vals) / max(len(vals), 1)


def mcnemar(rows, get_a, get_b):
    b01 = sum(1 for r in rows if get_a(r)["pred"] == r["gt"] and get_b(r)["pred"] != r["gt"])
    b10 = sum(1 for r in rows if get_a(r)["pred"] != r["gt"] and get_b(r)["pred"] == r["gt"])
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    k = min(b01, b10)
    p = min(1.0, sum(comb(n, j) for j in range(k + 1)) / 2**n * 2)
    return b01, b10, p


def main():
    base, aux = load(sys.argv[1]), load(sys.argv[2])
    common = sorted(set(base) & set(aux))
    B = [base[i] for i in common]
    A = [aux[i] for i in common]
    print(f"paired samples: {len(common)}\n")

    # ---- 1. isolation depth sweep --------------------------------------
    print("== Isolation sweep: block text->vision in layers [0..d) ==")
    print(f"{'cond':>8} | {'base acc':>8} {'aux acc':>8} | {'base NLL':>8} {'aux NLL':>8} | drop-vs-d0 (base/aux)")
    b0 = acc(B, lambda r: r["intact"])
    a0 = acc(A, lambda r: r["intact"])
    print(f"{'d0':>8} | {b0:8.3f} {a0:8.3f} | "
          f"{mean_nll(B, lambda r: r['intact']):8.3f} {mean_nll(A, lambda r: r['intact']):8.3f} |")
    for d in DEPTHS:
        key = f"iso_d{d}"
        ab, aa = acc(B, lambda r: r[key]), acc(A, lambda r: r[key])
        nb, na = mean_nll(B, lambda r: r[key]), mean_nll(A, lambda r: r[key])
        # within-arm McNemar vs intact: is the damage at depth d significant?
        _, _, pb = mcnemar(B, lambda r: r["intact"], lambda r: r[key])
        _, _, pa = mcnemar(A, lambda r: r["intact"], lambda r: r[key])
        print(f"{'d'+str(d):>8} | {ab:8.3f} {aa:8.3f} | {nb:8.3f} {na:8.3f} | "
              f"{ab-b0:+.3f} (p={pb:.3f}) / {aa-a0:+.3f} (p={pa:.3f})")
    sb = acc(B, lambda r: r["swap"])
    sa = acc(A, lambda r: r["swap"])
    print(f"{'swap':>8} | {sb:8.3f} {sa:8.3f} | "
          f"{mean_nll(B, lambda r: r['swap']):8.3f} {mean_nll(A, lambda r: r['swap']):8.3f} | (wrong-image floor)")

    # ---- 2. logit-lens arrival + image-swap sensitivity ------------------
    print("\n== Layer-k readout (final-norm + tied lm_head), answer position ==")
    print(f"{'layer':>6} | {'base acc':>8} {'aux acc':>8} | {'base NLL':>9} {'aux NLL':>9} | "
          f"{'swapΔNLL base':>13} {'swapΔNLL aux':>13}")
    for k in EXITS:
        gk = lambda r, k=k: r["intact"].get("exits", {}).get(str(k))
        gks = lambda r, k=k: r["swap"].get("exits", {}).get(str(k))
        ab, aa = acc(B, gk), acc(A, gk)
        nb, na = mean_nll(B, gk), mean_nll(A, gk)
        dsb = mean_nll(B, gks) - nb
        dsa = mean_nll(A, gks) - na
        print(f"{k:>6} | {ab:8.3f} {aa:8.3f} | {nb:9.3f} {na:9.3f} | {dsb:+13.3f} {dsa:+13.3f}")
    fb = mean_nll(B, lambda r: r["intact"])
    fa = mean_nll(A, lambda r: r["intact"])
    dfb = mean_nll(B, lambda r: r["swap"]) - fb
    dfa = mean_nll(A, lambda r: r["swap"]) - fa
    print(f"{'final':>6} | {b0:8.3f} {a0:8.3f} | {fb:9.3f} {fa:9.3f} | {dfb:+13.3f} {dfa:+13.3f}")

    # ---- 3. attention mass ----------------------------------------------
    print("\n== Attention mass text->vision (mean over heads & text queries) ==")
    Bm = [r for r in B if "attn_t2v" in r["intact"]]
    Am = [r for r in A if "attn_t2v" in r["intact"]]
    n_layers = len(Bm[0]["intact"]["attn_t2v"]) if Bm else 0
    print(f"   (base n={len(Bm)}, aux n={len(Am)})")
    print(f"{'layer':>6} | {'base t2v':>9} {'aux t2v':>9} | {'base last2v':>11} {'aux last2v':>11}")
    for li in range(n_layers):
        bt = sum(r["intact"]["attn_t2v"][li] for r in Bm) / len(Bm)
        at = sum(r["intact"]["attn_t2v"][li] for r in Am) / len(Am)
        bl = sum(r["intact"]["attn_last2v"][li] for r in Bm) / len(Bm)
        al = sum(r["intact"]["attn_last2v"][li] for r in Am) / len(Am)
        print(f"{li:>6} | {bt:9.4f} {at:9.4f} | {bl:11.4f} {al:11.4f}")


if __name__ == "__main__":
    main()
