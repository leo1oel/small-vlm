"""Figure 1 — per model, on one depth axis, the two pieces of evidence that matter:
  (1) GREEN ImageNet linear-probe top-1 of the image-token rep vs depth: for native (encoder-free)
      models it climbs from ~chance at layer 0 to the real-encoder ceiling band (DINO/CLIP/SigLIP)
      -> the model BUILDS a vision encoder internally. Encoder-VLMs are flat-high (encoder fed in).
  (2) The fusion READ-ONSET as a box-and-whisker along the same depth axis (q10/q25/q50/q75/q90 of
      the suffix-mean-ablation onset CDF): where the answer starts reading the image. It sits AFTER
      the probe has matured -> the encoder is built before fusion reads it.

The noisy per-layer "stream work" curves were dropped: they carry no clean cross-model rule and the
real, reference-free evidence is the linear-probe + fusion onset.

Reads imagenet_probe_results.json (probe) + results_sufpatch_<tag>.jsonl (fusion onset, via
patch_quants). Run with neo venv python (matplotlib).
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_analysis import load, patch_quants  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "neo_analysis"
OUT = Path(__file__).resolve().parents[1] / "neo_report" / "fig_mechanism.png"
plt.rcParams.update(
    {"figure.dpi": 150, "savefig.dpi": 150, "font.size": 10, "axes.spines.top": False}
)

PROBE_C, FUSE_C = "#1b7a3d", "#2f4d7a"
# (probe tag, sufpatch tag, label, group). Every model here has BOTH a linear-probe and a
# 200-sample fusion onset. Mono-InternVL is omitted: its fusion R0=0.013 (<0.05) is noise, so its
# read-onset is meaningless (it stays in the probe-only generality figure). ivl2/ivl4 have fusion
# but no probe.
MODELS = [
    ("gemma", "gemma12", "Gemma-4-12B", "native"),
    ("neo2b", "neo2b", "NEO1.0-2B", "native"),
    ("neo9b", "neo9b", "NEO1.0-9B", "native"),
    ("neo15_2b", "neo15_2b", "NEO1.5-2B", "native"),
    ("neo15_9b", "neo15_9b", "NEO1.5-9B", "native"),
    ("qwen", "qwen", "Qwen2.5-VL-7B", "encoder"),
    ("janus", "janus", "Janus-Pro-7B", "encoder"),
    ("ivl8", "ivl8", "InternVL3.5-8B", "encoder"),
    ("ivl30moe", "ivl30moe", "InternVL3.5-30B-A3B", "encoder"),
    ("gemma26moe", "gemma26moe", "Gemma-4-26B-MoE", "encoder"),
]


def probe_curve(R, t):
    e = R.get(t)
    if not e:
        return None
    pl = {int(k): v for k, v in e["per_layer"].items()}
    L = max(pl)
    xs = sorted(pl)
    return np.array([x / L for x in xs]), np.array([pl[x] for x in xs])


def fusion_quants(tag):
    p = ROOT / f"results_sufpatch_{tag}.jsonl"
    if not p.exists():
        return None
    q = patch_quants(load(p)).get("sufmeanabl")
    return q  # {q10,q25,q50,q75,q90} in relative depth, or None


def draw_fusion_box(ax, q, y0=0.045, h=0.05):
    """Horizontal box-and-whisker of the fusion read-onset along the depth (x) axis."""
    q10, q25, q50, q75, q90 = q["q10"], q["q25"], q["q50"], q["q75"], q["q90"]
    # whisker line q10..q90 with end caps
    ax.plot([q10, q90], [y0, y0], color=FUSE_C, lw=1.4, zorder=6)
    for xq in (q10, q90):
        ax.plot([xq, xq], [y0 - h * 0.45, y0 + h * 0.45], color=FUSE_C, lw=1.4, zorder=6)
    # box q25..q75
    ax.add_patch(
        Rectangle(
            (q25, y0 - h),
            q75 - q25,
            2 * h,
            facecolor=FUSE_C,
            alpha=0.30,
            edgecolor=FUSE_C,
            lw=1.3,
            zorder=7,
        )
    )
    # median q50
    ax.plot([q50, q50], [y0 - h, y0 + h], color=FUSE_C, lw=2.4, zorder=8)
    # faint full-height median guide to read against the probe curve
    ax.axvline(q50, color=FUSE_C, lw=1.0, ls=":", alpha=0.6, zorder=2)
    ax.text(q50, 0.875, "fusion\nread-onset", color=FUSE_C, fontsize=7.5, ha="center", va="top")


def main():
    R = json.load(open(ROOT / "imagenet_probe_results.json"))
    cz = [R[t]["peak_top1"] for t in ("dino", "clip", "siglip") if t in R]
    ceil_lo, ceil_hi = (min(cz), max(cz)) if cz else (0.74, 0.77)

    fig, axes = plt.subplots(4, 3, figsize=(16, 16))
    axes = axes.flatten()
    legend_idx = len(MODELS)
    for k, (pt, ft, label, grp) in enumerate(MODELS):
        ax = axes[k]
        ax.axhspan(ceil_lo, ceil_hi, color=PROBE_C, alpha=0.13, lw=0, zorder=0)
        ax.text(
            0.02,
            ceil_hi + 0.006,
            "real-encoder ceiling",
            color=PROBE_C,
            fontsize=7.5,
            style="italic",
            va="bottom",
        )
        ax.axhline(0.001, color="#aaa", ls=":", lw=0.9)
        pc = probe_curve(R, pt)
        if pc is not None:
            ax.plot(
                pc[0],
                pc[1],
                "-o",
                color=PROBE_C,
                lw=2.4,
                ms=5,
                zorder=5,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label="ImageNet linear-probe",
            )
        q = fusion_quants(ft)
        if q is not None:
            draw_fusion_box(ax, q)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.92)
        ax.set_xlabel("relative depth   ℓ / N")
        ax.set_ylabel("ImageNet linear-probe top-1", color=PROBE_C, fontsize=9)
        ax.tick_params(axis="y", labelcolor=PROBE_C)
        tag = "native — vision built INSIDE" if grp == "native" else "encoder-VLM — vision fed in"
        ax.set_title(
            f"{label}\n{tag}", fontsize=10.5, color="#b5530a" if grp == "native" else FUSE_C
        )

    for j in range(legend_idx, len(axes)):
        axes[j].axis("off")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    axes[legend_idx].legend(
        handles=[
            Line2D(
                [],
                [],
                color=PROBE_C,
                lw=2.4,
                marker="o",
                label="ImageNet linear-probe top-1\n(image rep → object identity)",
            ),
            Patch(
                fc=PROBE_C, alpha=0.2, label="real vision-encoder ceiling\n(DINO / CLIP / SigLIP)"
            ),
            Line2D([], [], color=FUSE_C, lw=2.4, label="fusion read-onset: median (q50)"),
            Patch(fc=FUSE_C, alpha=0.30, ec=FUSE_C, label="fusion IQR (q25–q75)"),
            Line2D([], [], color=FUSE_C, lw=1.4, label="fusion whiskers (q10–q90)"),
        ],
        loc="center",
        frameon=False,
        fontsize=10,
        title="Reading the figure",
        title_fontsize=12,
    )

    fig.suptitle(
        "Native VLMs build a vision encoder INSIDE the LLM: the GREEN ImageNet linear-probe of the image rep climbs from\n"
        "near-chance (layer 0) to the real-encoder ceiling — and the fusion read-onset (blue box-whisker) lands AFTER that, i.e.\n"
        "the encoder is built before the answer reads it. Encoder-VLMs (bottom row) get the encoder fed in (probe flat-high).",
        fontsize=12,
        y=1.0,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT, bbox_inches="tight")
    print(f"[fig] saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
