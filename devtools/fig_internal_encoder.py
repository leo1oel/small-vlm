"""P2 figure: does a native VLM grow an internal vision encoder?

Left: linear-predictivity R^2(depth) curves — native (enc-free) RISE from a low
layer-0 floor to encoder-grade; encoder-VLM controls are FLAT-HIGH from layer 0.
Right (if imagenet_probe_results.json present): ImageNet top-1 bars — native peak
layer vs real-encoder ceiling vs layer-0/pixel floor.

Usage: python devtools/fig_internal_encoder.py [predictivity.json] [encoder_for_curves]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "neo_analysis"
OUT = Path(__file__).resolve().parents[1] / "neo_report" / "fig_internal_encoder.png"
NATIVE_C = {"sail": "#1b9e77", "neo2bsft": "#66a61e", "neo9bsft": "#7570b3",
            "gemma": "#e6ab02", "mono": "#d95f02"}
ENCVLM_C = {"llava": "#999999", "qwen": "#666666"}
RAND_C = {"gemmarand": "#000000", "llavarand": "#cc3333"}
LABEL = {"sail": "SAIL-7B", "neo2bsft": "NEO-2B", "neo9bsft": "NEO-9B",
         "gemma": "Gemma-4-12B", "mono": "Mono-InternVL-2B",
         "llava": "LLaVA-1.5 (enc-VLM)", "qwen": "Qwen2.5-VL (enc-VLM)",
         "gemmarand": "Gemma-4 RANDOM-INIT", "llavarand": "LLaVA RANDOM-INIT"}


def main():
    pj = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "predictivity_n1000.json"
    ref = sys.argv[2] if len(sys.argv) > 2 else "siglip"
    d = json.load(open(pj))
    has_in = (ROOT / "imagenet_probe_results.json").exists()
    fig, axes = plt.subplots(1, 2 if has_in else 1, figsize=(13 if has_in else 7, 5))
    ax = axes[0] if has_in else axes

    for tag, ent in d.items():
        pe = ent["per_encoder"].get(ref)
        if pe is None:
            continue
        curve = np.array(pe["r2_curve"])
        x = np.linspace(0, 1, len(curve))
        role = ent["role"]
        if role == "native":
            c, style, lw = NATIVE_C.get(tag, "#d95f02"), "-o", 2.0
        elif role == "randinit":
            c, style, lw = RAND_C.get(tag, "#000000"), ":x", 1.6
        else:
            c, style, lw = ENCVLM_C.get(tag, "#bbbbbb"), "--s", 1.6
        ax.plot(x, curve, style, color=c, ms=3, lw=lw,
                label=LABEL.get(tag, tag), alpha=0.95 if role == "native" else 0.8)
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("relative depth (layer / total)")
    ax.set_ylabel(f"linear predictivity  R²  (layer → {ref})")
    ax.set_title("Native VLMs BUILD encoder-grade vision (rising);\nencoder-VLMs are flat-high (fed in)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)

    if has_in:
        ax2 = axes[1]
        r = json.load(open(ROOT / "imagenet_probe_results.json"))
        order = [t for t in ["dino", "clip", "siglip", "mono", "sail", "neo2bsft", "gemma",
                             "llava", "qwen"] if t in r]
        peaks = [r[t]["peak_top1"] for t in order]
        floors = [min(r[t]["per_layer"].values()) for t in order]
        is_enc = [t in ("dino", "clip", "siglip") for t in order]
        cols = []
        for t in order:
            if t in ("dino", "clip", "siglip"):
                cols.append("#377eb8")          # encoder ceiling
            elif t in ENCVLM_C:
                cols.append("#999999")          # enc-VLM control
            else:
                cols.append(NATIVE_C.get(t, "#d95f02"))  # native
        x = np.arange(len(order))
        first = True
        for i, t in enumerate(order):
            if is_enc[i]:
                ax2.bar(x[i], peaks[i], 0.55, color=cols[i])
            else:  # native + enc-VLM: show layer-0 floor vs peak layer
                lbl = ("layer-0 floor", "peak layer") if first else (None, None)
                ax2.bar(x[i] - 0.2, floors[i], 0.4, color="#dddddd", edgecolor="#999", label=lbl[0])
                ax2.bar(x[i] + 0.2, peaks[i], 0.4, color=cols[i], label=lbl[1])
                first = False
        ceil = np.mean([r[t]["peak_top1"] for t in ("dino", "clip", "siglip") if t in r])
        ax2.axhline(ceil, color="#377eb8", ls="--", lw=1, label=f"encoder ceiling ≈ {ceil:.2f}")
        ax2.set_xticks(x)
        ax2.set_xticklabels([LABEL.get(t, t).split(" ")[0] for t in order], rotation=40, ha="right", fontsize=8)
        ax2.set_ylabel("ImageNet-1k linear-probe top-1")
        ax2.set_title("Functional gold standard: native internal rep\nrises from pixel floor to ≥ encoder ceiling")
        ax2.legend(fontsize=7, loc="lower left")
        ax2.grid(alpha=0.25, axis="y")

    plt.tight_layout()
    plt.savefig(OUT, dpi=140)
    print(f"[fig] saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
