"""P1 generalization figure: fusion read-onset depth (sufmeanabl q50 + IQR) for every model,
showing mid-stack fusion generalizes across MoE / size / unified architectures.
Usage: python devtools/fig_p1_generalization.py   (run in an env with matplotlib, e.g. neo venv)
Output: neo_report/fig_p1_generalization.{png,pdf}
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_analysis import load, patch_quants  # noqa: E402

ROOT = Path("/mmfs1/gscratch/krishna/leoym/small-vlm/neo_analysis")
# (label, file, group) — top to bottom
MODELS = [
    ("LLaVA-1.5-7B (P0)", "results_sufpatch_llava.jsonl", "baseline"),
    ("Qwen2.5-VL-7B (P0)", "results_sufpatch_qwen.jsonl", "baseline"),
    ("NEO-2B (P0)", "results_sufpatch_neo2b.jsonl", "baseline"),
    ("SAIL-7B (P0)", "results_sufpatch_sail.jsonl", "baseline"),
    ("Janus-Pro-7B (unified)", "results_sufpatch_janus.jsonl", "unified"),
    ("Gemma-4-12B (dense)", "results_sufpatch_gemma12.jsonl", "gemma"),
    ("Gemma-4-26B-A4B (MoE)", "results_sufpatch_gemma26moe.jsonl", "gemma"),
    ("InternVL3.5-2B", "results_sufpatch_ivl2.jsonl", "internvl"),
    ("InternVL3.5-4B", "results_sufpatch_ivl4.jsonl", "internvl"),
    ("InternVL3.5-8B (dense)", "results_sufpatch_ivl8.jsonl", "internvl"),
    ("InternVL3.5-30B-A3B (MoE)", "results_sufpatch_ivl30moe.jsonl", "internvl"),
]
COL = {"baseline": "#888888", "unified": "#9b59b6", "gemma": "#e08e0b", "internvl": "#2e86ab"}

fig, ax = plt.subplots(figsize=(7.2, 5.6))
ax.axvspan(0.33, 0.66, color="#eef3f7", zorder=0)
ax.text(0.495, len(MODELS) + 0.4, "mid-stack [0.33, 0.66]", ha="center", fontsize=8, color="#5a7")
yt, yl = [], []
for i, (lab, f, g) in enumerate(MODELS):
    p = ROOT / f
    if not p.exists():
        print("skip (missing):", lab)
        continue
    q = patch_quants(load(p)).get("sufmeanabl")
    if not q:
        print("skip (no quants):", lab)
        continue
    y = len(MODELS) - i
    ax.errorbar(
        q["q50"],
        y,
        xerr=[[q["q50"] - q["q25"]], [q["q75"] - q["q50"]]],
        fmt="o",
        ms=8,
        color=COL[g],
        ecolor=COL[g],
        elinewidth=1.5,
        capsize=3,
        zorder=5,
    )
    yt.append(y)
    yl.append(lab)
    print("%-30s q50=%.2f [%.2f,%.2f]" % (lab, q["q50"], q["q25"], q["q75"]))
ax.set_yticks(yt)
ax.set_yticklabels(yl, fontsize=9)
ax.set_xlim(0, 0.8)
ax.set_ylim(0.3, len(MODELS) + 0.9)
ax.set_xlabel("fusion read-onset depth  (sufmeanabl q50, relative)")
ax.set_title("Mid-stack fusion generalizes across MoE / size / unified VLMs", fontsize=11)
from matplotlib.patches import Patch

ax.legend(handles=[Patch(color=COL[k], label=k) for k in COL], loc="lower right", fontsize=8)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(
        f"/mmfs1/gscratch/krishna/leoym/small-vlm/neo_report/fig_p1_generalization.{ext}", dpi=150
    )
print("saved neo_report/fig_p1_generalization.{png,pdf}")
