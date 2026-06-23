import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "neo_analysis"
VLMS = [("NEO1.0-2B-SFT", "neo2bsft", "free"), ("NEO1.0-9B-SFT", "neo9bsft", "free"),
        ("Gemma-4-12B", "gemma", "free"), ("SAIL-7B", "sail", "free"),
        ("LLaVA-1.5-7B", "llava", "enc"), ("Qwen2.5-VL-7B", "qwen", "enc")]
REFS = [("DINOv2", "dino"), ("CLIP-L", "clip"), ("SigLIP", "siglip")]


def _center(K):
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def cka(X, Y):
    # linear CKA via NxN Gram matrices (O(N^2 d), N=100 << d) — identical result
    Kc = _center(X @ X.T)
    Lc = _center(Y @ Y.T)
    return float((Kc * Lc).sum() / (np.sqrt((Kc * Kc).sum() * (Lc * Lc).sum()) + 1e-12))


def ld(tag):
    p = ROOT / f"cka_{tag}.npz"
    return np.load(p)["feats"].astype(np.float32) if p.exists() and p.stat().st_size else None


refs = {rn: ld(rt)[0] for rn, rt in REFS if ld(rt) is not None}
out = {}
for name, tag, fam in VLMS:
    f = ld(tag)
    if f is None:
        print(f"MISSING {tag}"); continue
    L = f.shape[0]
    out[name] = dict(family=fam, L=L)
    for rn, Y in refs.items():
        out[name][rn] = [cka(f[l], Y) for l in range(L)]
    c = out[name]["DINOv2"]
    print(f"{name:16}({fam}) L={L} DINO: L0={c[0]:.2f} q.25={c[int(.25*(L-1))]:.2f} "
          f"mid={c[L//2]:.2f} q.75={c[int(.75*(L-1))]:.2f} last={c[-1]:.2f} "
          f"peak={max(c):.2f}@rel{np.argmax(c)/(L-1):.2f} rise={max(c)-c[0]:+.2f}")
(ROOT / "cka_results.json").write_text(json.dumps(out, indent=1))
print("wrote cka_results.json (", len(out), "models )")
