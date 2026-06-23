"""Linear CKA between each VLM's per-layer image-token reps and frozen visual
encoders (DINOv2 / CLIP / SigLIP). Answers: at what decoder depth do encoder-
free models reach encoder-grade visual features, vs encoder-based models that
start there at layer 0? Reads neo_analysis/cka_*.npz, writes cka_results.json
and neo_report/fig_cka.png.

Run with neo venv python. Usage: python devtools/cka_compute.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "neo_analysis"
OUT = Path(__file__).resolve().parents[1] / "neo_report"

VLMS = [("NEO1.0-2B-SFT", "neo2bsft", "free"), ("NEO1.0-9B-SFT", "neo9bsft", "free"),
        ("Gemma-4-12B", "gemma", "free"), ("SAIL-7B", "sail", "free"),
        ("LLaVA-1.5-7B", "llava", "enc"), ("Qwen2.5-VL-7B", "qwen", "enc")]
REFS = [("DINOv2", "dino"), ("CLIP-L", "clip"), ("SigLIP", "siglip")]
FREE, ENC = "#1f77b4", "#ff7f0e"


def linear_cka(X, Y):
    # NxN Gram form (O(N^2 d), N=100 << d) — identical to the feature-space form
    n = X.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ (X @ X.T) @ H
    Lc = H @ (Y @ Y.T) @ H
    return float((Kc * Lc).sum() / (np.sqrt((Kc * Kc).sum() * (Lc * Lc).sum()) + 1e-12))


def load(tag):
    p = ROOT / f"cka_{tag}.npz"
    if not p.exists() or p.stat().st_size == 0:
        return None
    d = np.load(p)
    return d["feats"].astype(np.float32), d["idx"]


def main():
    refs = {}
    for rname, rtag in REFS:
        r = load(rtag)
        if r is not None:
            refs[rname] = r[0][0]  # (N,Hr)
    print("refs loaded:", list(refs.keys()))
    res = {}
    for name, tag, fam in VLMS:
        v = load(tag)
        if v is None:
            print(f"  MISSING {tag}")
            continue
        feats, idx = v  # (L+1,N,H)
        L = feats.shape[0]
        res[name] = dict(family=fam, L=L, idx=idx.tolist())
        for rname, Y in refs.items():
            res[name][rname] = [linear_cka(feats[l], Y) for l in range(L)]
        peak = {rn: (max(res[name][rn]), int(np.argmax(res[name][rn]))) for rn in refs}
        print(f"{name}: L={L} " + " ".join(
            f"{rn}:peak={pk[0]:.2f}@L{pk[1]}({pk[1]/(L-1):.2f})" for rn, pk in peak.items()))
    (ROOT / "cka_results.json").write_text(json.dumps(res, indent=1))

    if not refs:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nref = len(refs)
    fig, ax = plt.subplots(1, nref, figsize=(6 * nref, 5), squeeze=False)
    for j, (rname, _) in enumerate(refs.items()):
        a = ax[0][j]
        for name, tag, fam in VLMS:
            if name not in res or rname not in res[name]:
                continue
            c = res[name][rname]
            L = len(c)
            x = [l / (L - 1) for l in range(L)]
            a.plot(x, c, "-", color=FREE if fam == "free" else ENC, lw=1.6,
                   label=f"{name} (peak@{np.argmax(c)/(L-1):.2f})")
        a.set_title(f"CKA(image-token reps, {rname})")
        a.set_xlabel("relative decoder depth l/N"); a.set_ylabel("linear CKA")
        a.axvspan(0, 0.3, color="gray", alpha=.08)
        a.legend(fontsize=6.5)
    fig.suptitle("Where image-token reps reach encoder-grade features "
                 "(blue=encoder-free, orange=encoder)", y=1.02)
    fig.tight_layout(); fig.savefig(OUT / "fig_cka.png", dpi=130, bbox_inches="tight")
    print("wrote fig_cka.png")


if __name__ == "__main__":
    main()
