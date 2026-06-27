"""P0 triangulation figure: orthogonal residual-stream metric (sufmeanabl) vs attention-knockout phi.

For each anchor model we plot the attention-knockout fusion depth (x = phi q50, the existing study's
metric) against the activation-patch read-onset (y = sufmeanabl q50, the new value-pathway metric).
Points near the diagonal => the two mechanistically-independent causal methods agree on fusion depth.
Whiskers are the [q25,q75] interquartile range of each metric.

Usage: python devtools/fig_patch_triangulation.py
Outputs: neo_report/fig_patch_vs_knockout.{png,pdf}
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_analysis import knockout_phi, load, patch_quants  # noqa: E402

ROOT = Path("/mmfs1/gscratch/krishna/leoym/small-vlm/neo_analysis")
# (label, patch jsonl, knockout jsonl, marker color)
MODELS = [
    ("LLaVA-1.5", "results_sufpatch_llava.jsonl", "results_fusion_full_llava.jsonl", "#d1495b"),
    ("Qwen2.5-VL", "results_sufpatch_qwen.jsonl", "results_fusion_full_qwenvl.jsonl", "#2e86ab"),
    ("NEO-2B", "results_sufpatch_neo2b.jsonl", "results_fusion_full_neo_strat.jsonl", "#3c887e"),
    ("SAIL-7B", "results_sufpatch_sail.jsonl", "results_win_sail.jsonl", "#e08e0b"),
]


def main():
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot([0, 1], [0, 1], "--", color="#999", lw=1, zorder=1)  # agreement diagonal
    for label, pf, kf, c in MODELS:
        pp = ROOT / pf
        kp = ROOT / kf
        if not pp.exists():
            print(f"skip {label}: {pf} missing")
            continue
        pq = patch_quants(load(pp)).get("sufmeanabl")
        ko = knockout_phi(load(kp)) if kp.exists() else {}
        kq = ko.get("phi") or ko.get("onset")  # prefer prefix-phi; fall back to suffix-onset
        if not pq or not kq:
            print(f"skip {label}: missing quants (patch={bool(pq)} knockout={bool(kq)})")
            continue
        x, y = kq["q50"], pq["q50"]
        ax.errorbar(
            x,
            y,
            xerr=[[x - kq["q25"]], [kq["q75"] - x]],
            yerr=[[y - pq["q25"]], [pq["q75"] - y]],
            fmt="o",
            ms=9,
            color=c,
            ecolor=c,
            elinewidth=1.4,
            capsize=3,
            zorder=5,
        )
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(8, 6), fontsize=10, color=c)
        print(f"{label:12s} knockout-phi q50={x:.2f}  sufmeanabl q50={y:.2f}")
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel("attention-knockout fusion depth  $\\phi$  (q50)")
    ax.set_ylabel("activation-patch read-onset  (sufmeanabl q50)")
    ax.set_title("Two orthogonal causal metrics agree on fusion depth", fontsize=11)
    ax.set_aspect("equal")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(
            f"/mmfs1/gscratch/krishna/leoym/small-vlm/neo_report/fig_patch_vs_knockout.{ext}",
            dpi=150,
        )
    print("saved neo_report/fig_patch_vs_knockout.{png,pdf}")


if __name__ == "__main__":
    main()
