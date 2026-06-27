"""Figure: native VLMs grow a vision encoder internally — shown by ImageNet linear probing.
Per-layer linear-probe top-1 vs depth: native models climb from ~chance (layer 0) to the real
vision-encoder ceiling (DINO/CLIP/SigLIP); encoder-VLMs are flat-high (encoder fed in). This is the
functional, reference-free evidence (replaces the predictivity/similarity argument).

Reads neo_analysis/imagenet_probe_results.json. Run with neo venv python. Usage: python devtools/fig_vision_encoder_probe.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "neo_analysis"
OUT = Path(__file__).resolve().parents[1] / "neo_report" / "fig_vision_encoder_probe.png"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# native = encoder-FREE (no separate vision tower). Gemma-4-12B is the "unified" variant
# (no deep encoder; L0 image rep is near-chance and climbs like NEO/SAIL). Gemma-4-26B-MoE is
# NOT here: it ships a full SigLIP encoder (gemma4_vision, hidden=1152) -> encoder group.
NATIVE = [
    ("sail", "SAIL-7B"),
    ("gemma", "Gemma-4-12B"),
    ("mono", "Mono-InternVL-2B"),
    ("neo2b", "NEO1.0-2B"),
    ("neo9b", "NEO1.0-9B"),
    ("neo15_2b", "NEO1.5-2B"),
    ("neo15_9b", "NEO1.5-9B"),
]
# encoder-VLM = ships an external vision tower (SigLIP / InternViT / etc.)
ENCVLM = [
    ("qwen", "Qwen2.5-VL-7B"),
    ("llava", "LLaVA-1.5-7B"),
    ("janus", "Janus-Pro-7B"),
    ("ivl8", "InternVL3.5-8B"),
    ("ivl30moe", "InternVL3.5-30B-A3B"),
    ("gemma26moe", "Gemma-4-26B-MoE"),
]
# Excluded: LLaVA-OneVision-1.5 — its pixel-shuffle merger makes the LLM-side image tokens
# share a dominant DC direction, so the mean-pooled rep probes flat ~0.15 (robust to L2-norm /
# thumbnail-only variants). An honest outlier of the readout, not encoder evidence; see scratch.md.
CEIL = ["dino", "clip", "siglip"]


def curve(r, t):
    e = r.get(t)
    if not e:
        return None
    pl = {int(k): v for k, v in e["per_layer"].items()}
    L = max(pl)
    xs = sorted(pl)
    return np.array([x / L for x in xs]), np.array([pl[x] for x in xs])


def main():
    r = json.load(open(ROOT / "imagenet_probe_results.json"))
    fig, ax = plt.subplots(figsize=(9, 6))
    # real-encoder ceiling band
    cz = [r[t]["peak_top1"] for t in CEIL if t in r]
    if cz:
        ax.axhspan(min(cz), max(cz), color="#2ca25f", alpha=0.13, zorder=0, lw=0)
        ax.text(
            0.015,
            max(cz) + 0.005,
            "real vision encoders (DINO / CLIP / SigLIP) — the ceiling",
            color="#1b7a3d",
            fontsize=9,
            style="italic",
            va="bottom",
        )
    ax.axhline(0.001, color="#999", ls=":", lw=1)
    ax.text(0.99, 0.012, "chance (1/1000)", color="#999", fontsize=8, ha="right")

    for grp, models, cmap, mk in [
        ("native", NATIVE, cm.Oranges, "o"),
        ("encvlm", ENCVLM, cm.Blues, "s"),
    ]:
        present = [(t, lb) for t, lb in models if curve(r, t) is not None]
        for j, (t, lb) in enumerate(present):
            x, y = curve(r, t)
            c = cmap(0.45 + 0.5 * (j / max(len(present) - 1, 1)))
            ax.plot(
                x,
                y,
                "-" + mk,
                color=c,
                lw=2.2,
                ms=5,
                label=lb,
                zorder=3,
                markeredgecolor="white",
                markeredgewidth=0.8,
            )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.88)
    ax.set_xlabel("relative depth   ℓ / N   (0 = input embedding, 1 = output)")
    ax.set_ylabel("ImageNet-1k linear-probe top-1  (frozen image rep + single linear head)")
    ax.set_title(
        "Native VLMs grow a vision encoder INSIDE: image-token linear-probe climbs from\n"
        "near-chance to the real-encoder ceiling.  Encoder-VLMs are flat-high (encoder fed in).",
        fontsize=11.5,
    )
    leg1 = ax.legend(
        [h for h in ax.lines if h.get_label() in dict(NATIVE).values()],
        [l for l in dict(NATIVE).values() if any(h.get_label() == l for h in ax.lines)],
        title="native — built inside",
        loc="center right",
        frameon=False,
        fontsize=8.5,
        title_fontsize=9,
    )
    ax.add_artist(leg1)
    ax.legend(
        [h for h in ax.lines if h.get_label() in dict(ENCVLM).values()],
        [l for l in dict(ENCVLM).values() if any(h.get_label() == l for h in ax.lines)],
        title="encoder-VLM — fed in",
        loc="lower right",
        frameon=False,
        fontsize=8.5,
        title_fontsize=9,
    )
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"[fig] saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
