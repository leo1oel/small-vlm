"""Fusion-WINDOW analysis: combine prefix-blocking (falling) and suffix-blocking
(rising) curves per model. Answers "when does fusion START" (suffix rise) vs
"when must it finish" (prefix fall) — the user's onset critique.

suffix file fields: suf[k] = block [k..N) (first k layers free), k=0..N-1.
prefix from results_fusion_full_* (cost[k] = block [0..k+1)).

Run with neo venv python. Usage: python devtools/window_analysis.py [out.png]
"""

import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1] / "neo_analysis"
OUT = Path(__file__).resolve().parents[1] / "neo_report"

# name -> (family, prefix_file or None(use win file), win_file)
MODELS = {
    "LLaVA-1.5-7B":   ("enc",  "results_fusion_full_llava.jsonl",  "results_win_llava.jsonl"),
    "LLaVA-NeXT-7B":  ("enc",  None,                               "results_win_llavanext.jsonl"),
    "OneVision-7B":   ("enc",  None,                               "results_win_onevision.jsonl"),
    "Qwen2.5-VL-7B":  ("enc",  "results_fusion_full_qwenvl.jsonl", "results_win_qwenvl.jsonl"),
    "NEO1.0-2B-SFT":  ("free", "results_fusion_full_neo_strat.jsonl", "results_win_neo2b.jsonl"),
    "NEO1.0-9B-SFT":  ("free", "results_fusion_full_neo_NEO1_0-9B-SFT.jsonl", "results_win_neo9b.jsonl"),
    "Gemma-4-12B":    ("free", "results_fusion_full_gemma4_strat.jsonl", "results_win_gemma.jsonl"),
    "SAIL-7B":        ("free", "results_fusion_full_sail.jsonl",   "results_win_sail.jsonl"),
}


def load(path):
    p = ROOT / path
    if not p.exists() or p.stat().st_size == 0:
        return None
    return [json.loads(l) for l in open(p) if l.strip() and '"skip"' not in l]


def curve(rows, key):
    causal = [r for r in rows if key in r]
    if len(causal) < 20:
        return None
    N = len(causal[0][key])

    def acc(g):
        return mean(g(r)["pred"] == r["gt"] for r in causal)
    R0 = acc(lambda r: r["intact"]) - acc(lambda r: r["swap"])
    if R0 <= 0.02:
        return None
    ret = [acc(lambda r, d=d: r[key][d]) - acc(lambda r, d=d: r[key + "_null"][d])
           for d in range(N)]
    return dict(N=N, R0=R0, rn=[x / R0 for x in ret], nc=len(causal),
                intact=acc(lambda r: r["intact"]), swap=acc(lambda r: r["swap"]))


def first_cross(rn, th, above, persist=2):
    N = len(rn)
    for i in range(N - persist + 1):
        ok = all((rn[i + j] >= th if above else rn[i + j] < th) for j in range(persist))
        if ok:
            return i / N if above else (i + 1) / N
    return 1.0


def lastrow_analysis():
    """Compare all-text-rows prefix blocking vs answer-row-only blocking.
    If a model's early fusion necessity flows image->question-tokens (gist) and
    only later question->answer, the lastrow curve stays flat early while the
    alltext curve drops (Zhang et al. 2411.18620 two-stage picture)."""
    pairs = {"LLaVA-1.5-7B": ("results_fusion_full_llava.jsonl", "results_winlr_llava.jsonl"),
             "Qwen2.5-VL-7B": ("results_fusion_full_qwenvl.jsonl", "results_winlr_qwenvl.jsonl")}
    out = {}
    for name, (afile, lfile) in pairs.items():
        arows, lrows = load(afile), load(lfile)
        if not arows or not lrows:
            continue
        ca = curve(arows, "cost")
        cl = curve(lrows, "cost")
        if not ca or not cl:
            continue
        out[name] = dict(alltext=ca, lastrow=cl)
        print(f"\n[lastrow] {name}: alltext o80={first_cross(ca['rn'],0.8,False):.2f} "
              f"lastrow o80={first_cross(cl['rn'],0.8,False):.2f} "
              f"(lastrow floor: min={min(cl['rn']):.2f} — flat curve means the direct "
              f"image->answer path is NOT load-bearing)")
    return out


def main():
    res = {}
    print(f"{'model':14} {'fam':4} | suffix-rise: 20%   50%   80%  | prefix-fall: o80   c20  | window(50%rise->o80)")
    print("-" * 105)
    for name, (fam, pfile, wfile) in MODELS.items():
        wrows = load(wfile)
        suf = curve(wrows, "suf") if wrows else None
        pre = None
        if pfile:
            prows = load(pfile)
            pre = curve(prows, "cost") if prows else None
        elif wrows:
            pre = curve(wrows, "cost")
        rec = dict(family=fam)
        s20 = s50 = s80 = po80 = pc20 = None
        if suf:
            s20 = first_cross(suf["rn"], 0.2, True)
            s50 = first_cross(suf["rn"], 0.5, True)
            s80 = first_cross(suf["rn"], 0.8, True)
            rec["suf"] = suf
        if pre:
            po80 = first_cross(pre["rn"], 0.8, False)
            pc20 = first_cross(pre["rn"], 0.2, False)
            rec["pre"] = pre
        res[name] = rec
        f = lambda x: f"{x:.2f}" if x is not None else " -- "
        print(f"{name:14} {fam:4} |        {f(s20)}  {f(s50)}  {f(s80)} |        {f(po80)}  {f(pc20)} |  "
              f"{f((po80 - s50) if (po80 is not None and s50 is not None) else None)}"
              f"   (suf nc={suf['nc'] if suf else 0})")
    (ROOT / "window_results.json").write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items()} for k, v in res.items()}, indent=1))

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    have = [(k, v) for k, v in res.items() if "suf" in v]
    if not have:
        return
    ncol = 4
    nrow = (len(have) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow), squeeze=False)
    for ax_i, (name, v) in enumerate(have):
        a = axes[ax_i // ncol][ax_i % ncol]
        c = "#1f77b4" if v["family"] == "free" else "#ff7f0e"
        s = v["suf"]
        xs = [i / s["N"] for i in range(s["N"])]
        a.plot(xs, s["rn"], "-", color=c, lw=2, label="suffix: signal ALREADY fused\nby depth d (block [d..N))")
        if "pre" in v:
            p = v["pre"]
            xp = [(i + 1) / p["N"] for i in range(p["N"])]
            a.plot(xp, p["rn"], "--", color=c, lw=1.4, alpha=.7,
                   label="prefix: signal still extractable\nafter blocking [0..d)")
        a.axhline(0.5, color="k", lw=.5, ls=":")
        a.set_title(f"{name}  (R0={s['R0']:+.2f})", fontsize=10)
        a.set_xlim(0, 1); a.set_ylim(-0.25, 1.3)
        a.set_xlabel("relative depth d/N", fontsize=8)
        if ax_i % ncol == 0:
            a.set_ylabel("retained / R0", fontsize=8)
        if ax_i == 0:
            a.legend(fontsize=6.5, loc="upper left")
    for j in range(len(have), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("Fusion WINDOW: solid rising = cumulative fused signal (onset); "
                 "dashed falling = late-fusion capacity (completion)", fontsize=11)
    fig.tight_layout()
    out = sys.argv[1] if len(sys.argv) > 1 else str(OUT / "fig_window.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)

    lr = lastrow_analysis()
    if lr:
        fig2, axs = plt.subplots(1, len(lr), figsize=(5.5 * len(lr), 4.2), squeeze=False)
        for i, (name, v) in enumerate(lr.items()):
            a = axs[0][i]
            for key, sty, lab in (("alltext", "-", "block ALL text rows -> image"),
                                  ("lastrow", "--", "block ONLY answer row -> image")):
                c = v[key]
                x = [(k + 1) / c["N"] for k in range(c["N"])]
                a.plot(x, c["rn"], sty, lw=2, label=f"{lab} (R0={c['R0']:+.2f})")
            a.axhline(1.0, color="k", lw=.5, ls=":")
            a.set_title(name); a.set_xlabel("blocked depth d/N"); a.set_ylim(-0.3, 1.4)
            a.legend(fontsize=7)
        axs[0][0].set_ylabel("retained/R0")
        fig2.suptitle("Pathway decomposition: image->question-tokens vs image->answer directly")
        fig2.tight_layout()
        fig2.savefig(OUT / "fig_lastrow.png", dpi=130)
        print("wrote fig_lastrow.png")


if __name__ == "__main__":
    main()
