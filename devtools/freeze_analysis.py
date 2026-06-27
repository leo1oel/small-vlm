"""Analyze freeze-probe results: per model, accuracy and image-signal R0 as a
function of freeze depth k, for freeze_img vs freeze_txt.

Reading: if freeze_img@k keeps intact-acc and R0 at baseline, layers [0..k)
do nothing essential to the image stream (encoder-model prediction). If
freeze_txt@k is harmless, those layers do nothing essential to text (NEO
pre-Buffer prediction).

Run with neo venv python. Usage: python devtools/freeze_analysis.py
"""

import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1] / "neo_analysis"
OUT = Path(__file__).resolve().parents[1] / "neo_report"

MODELS = {
    "LLaVA-1.5-7B": ("enc", "results_freeze_llava.jsonl"),
    "Qwen2.5-VL-7B": ("enc", "results_freeze_qwenvl.jsonl"),
    "Gemma-4-12B": ("free", "results_freeze_gemma.jsonl"),
    "NEO1.0-2B-SFT": ("free", "results_freeze_neo2b.jsonl"),
    "SAIL-7B": ("free", "results_freeze_sail.jsonl"),
}


def load(path):
    p = ROOT / path
    if not p.exists() or p.stat().st_size == 0:
        return None
    return [json.loads(l) for l in open(p) if l.strip() and '"skip"' not in l]


def main():
    res = {}
    for name, (fam, f) in MODELS.items():
        rows = load(f)
        if not rows:
            print(f"{name}: MISSING")
            continue
        causal = [r for r in rows if "freeze_img" in r]
        if len(causal) < 20:
            print(f"{name}: nc={len(causal)} (too few)")
            continue
        ks = causal[0]["ks"]
        causal = [r for r in causal if r["ks"] == ks]
        N = {
            "LLaVA-1.5-7B": 32,
            "Qwen2.5-VL-7B": 28,
            "Gemma-4-12B": 48,
            "NEO1.0-2B-SFT": 40,
            "SAIL-7B": 32,
        }[name]

        def acc(g):
            return mean(g(r)["pred"] == r["gt"] for r in causal)

        intact = acc(lambda r: r["intact"])
        swap = acc(lambda r: r["swap"])
        R0 = intact - swap
        rec = dict(family=fam, ks=ks, N=N, nc=len(causal), intact=intact, swap=swap, R0=R0)
        for mode in ("img", "txt"):
            a = [acc(lambda r, j=j: r[f"freeze_{mode}"][j]) for j in range(len(ks))]
            an = [acc(lambda r, j=j: r[f"freeze_{mode}_null"][j]) for j in range(len(ks))]
            rec[f"acc_{mode}"] = a
            rec[f"R0_{mode}"] = [a[j] - an[j] for j in range(len(ks))]
        res[name] = rec
        print(
            f"\n{name} ({fam}) nc={len(causal)} N={N} | baseline intact={intact:.2f} R0={R0:+.2f}"
        )
        print("  k/N        : " + " ".join(f"{k / N:5.2f}" for k in ks))
        print("  frzIMG acc : " + " ".join(f"{x:5.2f}" for x in rec["acc_img"]) + "   (vs intact)")
        print(
            "  frzIMG R0  : " + " ".join(f"{x:+5.2f}" for x in rec["R0_img"]) + f"   (vs {R0:+.2f})"
        )
        print("  frzTXT acc : " + " ".join(f"{x:5.2f}" for x in rec["acc_txt"]))
        print("  frzTXT R0  : " + " ".join(f"{x:+5.2f}" for x in rec["R0_txt"]))
    (ROOT / "freeze_results.json").write_text(json.dumps(res, indent=1))

    if not res:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(res), figsize=(4.4 * len(res), 4.0), squeeze=False)
    for i, (name, v) in enumerate(res.items()):
        a = axes[0][i]
        x = [k / v["N"] for k in v["ks"]]
        a.axhline(v["R0"], color="k", lw=1, ls=":", label=f"baseline R0={v['R0']:+.2f}")
        a.plot(x, v["R0_img"], "-o", color="#d62728", label="freeze IMAGE [0..k)")
        a.plot(x, v["R0_txt"], "-s", color="#9467bd", label="freeze TEXT [0..k)")
        a.set_title(f"{name} ({v['family']})", fontsize=10)
        a.set_xlabel("freeze depth k/N")
        a.set_ylim(-0.1, max(0.45, v["R0"] + 0.1))
        if i == 0:
            a.set_ylabel("usable image signal R0(k)")
        a.legend(fontsize=7)
    fig.suptitle(
        "FREEZE test: image signal surviving when one stream's residual is held at "
        "layer-0 values through [0..k)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_freeze.png", dpi=130)
    print("\nwrote fig_freeze.png")


if __name__ == "__main__":
    main()
