"""Linear predictivity (P2, Claim B): can a native VLM layer's image representation
be linearly mapped onto a real frozen vision encoder's representation?

For each model M, layer ell, encoder E: fit a cross-validated ridge regression
  X_{M,ell} (N, H_M)  ->  Y_E (N, H_E)
and report the held-out (5-fold) R^2 = fraction of the encoder representation that is
linearly recoverable from the native layer representation.

Rigor:
  - Per-fold fit of StandardScaler + PCA(X)->kx and PCA(Y)->ky (target reduced to its
    top-variance subspace): NO leakage (scalers/PCAs fit on train fold only), and the
    held-out R^2 cannot be inflated by overfitting (n > kx).
  - shuffled_floor (permuted Y) is the true null: should be ~0. Guards against a
    spuriously high baseline ("everything predicts everything").
  - R^2 reported as variance_weighted over the encoder's PCA components.

Discriminative test (the heart of Claim B):
  - native enc-free model (NEO/SAIL/Gemma/Mono): R^2(depth) RISES from a low layer-0 floor
    to encoder-grade -> the early/internal layers BUILD an encoder-like representation.
  - encoder-VLM control (LLaVA/Qwen): R^2(depth) is FLAT-HIGH from layer 0 (the encoder
    fed LLM-ready features in; nothing to build).
  - layer-0 / shuffled-label: low floor.

Reads neo_analysis/cka_<tag>.npz {feats (L+1,N,H), idx}. CPU-only. Saves predictivity_results.json.

Usage: python devtools/predictivity_compute.py [out.json] [tag1,tag2,...]
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "2")
from joblib import Parallel, delayed  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(os.environ.get("CKA_DIR", str(Path(__file__).resolve().parents[1] / "neo_analysis")))

# kind: "native" (enc-free; expect rising), "encvlm" (encoder-based control; expect flat-high)
MODELS = {
    "neo2bsft": "native",
    "neo9bsft": "native",
    "sail": "native",
    "gemma": "native",
    "mono": "native",
    "gemma26moe": "native",
    "llava": "encvlm",
    "qwen": "encvlm",
    "ivl2": "encvlm",
    "ivl4": "encvlm",
    "ivl8": "encvlm",
    "ivl30moe": "encvlm",
    "janus": "encvlm",
    "gemmarand": "randinit",
    "llavarand": "randinit",  # causal control (untrained weights)
}
ENCODERS = ["dino", "clip", "siglip"]
PCA_KX = 256  # input subspace
PCA_KY = 128  # target (encoder) subspace
N_FOLDS = 5
ALPHAS = np.logspace(0, 5, 6)
N_JOBS = int(os.environ.get("PRED_JOBS", "8"))


def load(tag):
    f = ROOT / f"cka_{tag}.npz"
    if not f.exists():
        return None
    return np.load(f)["feats"].astype(np.float32)  # (L+1 or 1, N, H)


def _fold_r2(Xtr, Ytr, Xte, Yte, kx, ky):
    sx = StandardScaler().fit(Xtr)
    Xtr, Xte = sx.transform(Xtr), sx.transform(Xte)
    px = PCA(n_components=kx, random_state=0).fit(Xtr)
    Xtr, Xte = px.transform(Xtr), px.transform(Xte)
    sy = StandardScaler().fit(Ytr)
    Ytr, Yte = sy.transform(Ytr), sy.transform(Yte)
    py = PCA(n_components=ky, random_state=0).fit(Ytr)
    Ytr, Yte = py.transform(Ytr), py.transform(Yte)
    r = RidgeCV(alphas=ALPHAS).fit(Xtr, Ytr)
    return r2_score(Yte, r.predict(Xte), multioutput="variance_weighted")


def cv_r2(X, Y):
    """Held-out 5-fold R^2 of a (StandardScaler+PCA)->RidgeCV map X->Y, per-fold fit."""
    n = X.shape[0]
    kx = min(PCA_KX, n - n // N_FOLDS - 1, X.shape[1])
    ky = min(PCA_KY, n - n // N_FOLDS - 1, Y.shape[1])
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    sc = [_fold_r2(X[tr], Y[tr], X[te], Y[te], kx, ky) for tr, te in kf.split(X)]
    return float(np.mean(sc)), float(np.std(sc))


def model_encoder_curve(feats, Y, do_shuffle_at):
    """All-layer R^2 curve for one (model, encoder). do_shuffle_at = layer index to also
    run a row-permuted null at (or None)."""
    L = feats.shape[0]
    curve = [cv_r2(feats[ell], Y) for ell in range(L)]
    r2 = [c[0] for c in curve]
    sd = [c[1] for c in curve]
    shuf = None
    if do_shuffle_at is not None:
        rng = np.random.default_rng(0)
        Yp = Y[rng.permutation(Y.shape[0])]
        shuf = cv_r2(feats[do_shuffle_at], Yp)[0]
    return r2, sd, shuf


def main():
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "predictivity_results.json"
    only = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
    enc = {e: load(e) for e in ENCODERS}
    enc = {e: v[0] for e, v in enc.items() if v is not None}
    print(f"[pred] encoders: {[(e, v.shape) for e, v in enc.items()]} jobs={N_JOBS}", flush=True)

    models = {t: r for t, r in MODELS.items() if (only is None or t in only)}
    feats_all = {t: load(t) for t in models}
    feats_all = {t: f for t, f in feats_all.items() if f is not None}
    print(f"[pred] models: {[(t, feats_all[t].shape) for t in feats_all]}", flush=True)

    # peak-layer guess for the shuffle null = 60% depth (cheap; just needs a real layer)
    tasks = []
    for t in feats_all:
        L = feats_all[t].shape[0]
        for e in enc:
            tasks.append((t, e, int(0.6 * (L - 1))))
    out = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(model_encoder_curve)(feats_all[t], enc[e], sh) for (t, e, sh) in tasks
    )

    results = {}
    for (t, e, _), (r2, sd, shuf) in zip(tasks, out):
        L = feats_all[t].shape[0]
        ent = results.setdefault(
            t,
            {
                "role": models[t],
                "n_layers": L,
                "n_images": int(feats_all[t].shape[1]),
                "per_encoder": {},
            },
        )
        curve = np.array(r2)
        peak_l = int(curve.argmax())
        ent["per_encoder"][e] = {
            "r2_curve": curve.round(4).tolist(),
            "r2_std": np.array(sd).round(4).tolist(),
            "peak_r2": float(curve[peak_l]),
            "peak_layer": peak_l,
            "peak_depth_frac": round(peak_l / (L - 1), 3),
            "floor_layer0_r2": round(float(curve[0]), 4),
            "shuffled_floor_r2": round(float(shuf), 4),
            "rise": round(float(curve[peak_l]) - float(curve[0]), 4),
        }
        print(
            f"[pred] {t:10s} {e:7s} peak={curve[peak_l]:.3f}@L{peak_l}/{L - 1} "
            f"floor={curve[0]:.3f} shuf={shuf:.3f} rise={curve[peak_l] - curve[0]:+.3f}",
            flush=True,
        )
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[pred] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
