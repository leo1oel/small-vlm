# P2 — Claim B completeness: is a native VLM's internal "vision encoder" real?

*Design spec, 2026-06-18. Tests the central unifying claim: **native VLMs grow an internal vision
encoder via generative pre-training** — i.e. the parameters that do the encoder's job produce visual
representations that are geometrically similar AND functionally equivalent to a real frozen encoder.
This completes Claim B (vision built before fusion), which P1 only addressed for Claim A (fusion
depth). Companion to `CROSS_VALIDATION.md`, `GENERALIZATION_RESULTS.md`, and the P1 spec.*

---

## 0. Why — the gap P1 left

P1 generalized **Claim A** (mid-stack fusion) across 11 models + benchmarks. It did **not** verify
**Claim B** for new models. Current Claim-B evidence is only **CKA** (geometric similarity) on
NEO/SAIL/Gemma/LLaVA/Qwen (`cka_*.npz`, `fig_prebuffer_cka`), which (a) wasn't extended to new
native models, (b) has **no functional (linear-probing) evidence**, and (c) CKA-to-DINO is
confounded (raw pixels already ~0.6). So the claim "native internal params = a trained vision
encoder" is **not yet established**. This spec makes it complete and rigorous.

**The unifying story we are testing:** native VLM = an internal encoder (early layers / per-layer
visual experts build the visual representation) + the LLM (mid-stack fusion reads it). If the
internal representation is encoder-grade, Claim A (where fusion happens) and Claim B (vision built
first) become one mechanism.

## 1. Goals & success criteria

- **G1 — Geometric (CKA):** extend the existing per-layer CKA to the new native model(s).
- **G2 — Functional, predictivity (PRIMARY, new):** show a native layer's image representation can be
  linearly mapped to a real encoder's representation with high cross-validated R². **Discriminative
  test:** native models show a *rising* R²(depth) curve (build-up to encoder-grade); encoder-based
  VLMs (LLaVA/Qwen) show a *flat-high* curve (the encoder fed it in, nothing to build). That
  shape difference is the direct evidence of an internally-grown encoder.
- **G3 — Functional, downstream (gold standard, new):** ImageNet linear probing — a native model's
  peak-layer representation reaches top-1 accuracy comparable to the real encoder's (and to the
  encoder-VLM's internal layers), not to raw pixels.
- **Success:** native models exhibit (a) rising CKA & predictivity to encoder-grade by some internal
  depth, and (b) ImageNet probe accuracy at that depth approaching the real-encoder ceiling — while
  the encoder-VLM controls stay flat-high and the raw-pixel/layer-0 floor stays low. If native peak
  ≈ encoder and ≫ floor, the "internal encoder" claim is established.

Non-goal: re-deriving Claim A; training any model; SOTA ImageNet accuracy (we need *relative*
comparison under one fixed protocol, not absolute SOTA).

## 2. Metrics (three, weak→strong)

### 2.1 CKA (geometric, existing) — extend only
Linear CKA between each model's per-layer mean-pooled image-token reps and a frozen encoder's
patch-mean rep, over a fixed image set. Reuse `cka_extract.py` + `cka_compute.py`. Extend to
Mono-InternVL. Keep the **calibration** point (LLaVA image tokens vs its own CLIP ≈ 0.89) and the
**caveat** (CKA-to-DINO confounded). CKA is supporting, not decisive.

### 2.2 Linear predictivity (PRIMARY, new)
For native model M, layer ℓ, encoder E: fit a **cross-validated ridge regression**
`X_{M,ℓ} (n, H_M) → Y_E (n, H_E)` and report the held-out **R²** (variance of the encoder rep
linearly explained by the native layer rep). n = **1000 fixed images** (extended from CKA's 100 —
larger n stabilizes the regression; H is large so use 5-fold CV ridge with α chosen by inner CV, and
optionally PCA-reduce X to ~256 dims first to keep n > d).
- **Curves:** R²(ℓ, E) per model. Report peak R² and peak depth.
- **Discriminative control:** native = rising R²(depth); encoder-VLM (LLaVA/Qwen) = flat-high;
  layer-0 / raw-pixel = low floor.
- Three encoders E ∈ {DINOv2, CLIP, SigLIP} → robustness to the reference choice.
- *Why predictivity over CKA:* it directly answers "are the two representations interchangeable"
  (can one be linearly recovered from the other) rather than just "are they geometrically aligned".

### 2.3 ImageNet linear probing (gold-standard functional, new)
Dataset: **`mrm8488/ImageNet1K-val`** (39.3k images, `image` + `label` 1000-class ClassLabel,
5.2 GB parquet, **not gated**). Protocol (identical for every representation source):
- Extract the mean-pooled image rep at the model's **peak layer** (peak from §2.2/§2.1) for all 39.3k
  images → (39.3k, H).
- Split per-class ~80/20 (stratified) → train a multinomial **logistic-regression** linear head
  (sklearn `LogisticRegression` or a torch linear + LBFGS) → report test **top-1 accuracy**.
- **Representation sources compared under the same protocol:** native peak layer (Mono/NEO/SAIL/
  Gemma); real encoders DINOv2/CLIP/SigLIP (the **ceiling**); encoder-VLM LLaVA/Qwen internal peak
  layer (control — should already be encoder-grade); native **layer-0 / raw-pixel** (the **floor**).
- **Success:** native peak acc ≈ encoder ceiling and ≫ floor.

## 3. Models & controls

| Role | Models | status |
|---|---|---|
| **Native (subject)** | Mono-InternVL-2B (= this repo's `feat/visual-ffn-expert` arch; **new custom dir**), NEO-2B/9B, SAIL-7B, Gemma-4-12B | NEO/SAIL/Gemma have CKA; all need predictivity + ImageNet |
| **Encoder-VLM (upper-bound control)** | LLaVA-1.5, Qwen2.5-VL | internal layers should be flat-high (encoder fed in) |
| **Real encoders (ceiling)** | DINOv2, CLIP-L/14, SigLIP | already extracted (`cka_{dino,clip,siglip}.npz`) |
| **Floor** | layer-0 / raw-pixel-patch rep | sanity lower bound |

Mono-InternVL is the headline native subject (it's *your* architecture and the most direct Claim-B
test: a per-layer visual FFN expert — does it build encoder-grade features?).

## 4. Rigor (how we avoid a wrong conclusion)
- Predictivity uses **held-out CV R²** (never train-set R²) → no overfitting illusion with large H.
- **Discriminative controls** are the heart: native *rising* vs encoder-VLM *flat-high* vs floor
  *low*. A bare "native R² is high" is not enough — it must rise from a low floor and the encoder-VLM
  must stay flat, or the metric isn't measuring "building an encoder."
- **Three reference encoders** (no single-reference bias); report per-encoder + agreement.
- **CKA caveat stated**; predictivity + ImageNet are the load-bearing evidence.
- **Calibration**: LLaVA→its-own-CLIP predictivity/CKA must be high (known sanity).
- One **fixed protocol** for ImageNet probing across all representation sources (same split, same
  head, same preprocessing) so the comparison is fair.

## 5. Engineering plan

Files:
- Modify `devtools/cka_extract.py` — support N=1000; add `internvl`/`janus`/`gemma4moe` kinds if reused;
  it already saves per-layer mean-pooled reps → also the input to predictivity (no separate extractor
  needed). Encoder reps already exist; re-extract at N=1000 for row-alignment.
- Create `devtools/predictivity_compute.py` — load `cka_*.npz` (N=1000), per (model-layer, encoder)
  CV-ridge R², save `predictivity_results.json` + curves.
- Create `devtools/imagenet_probe.py` — (a) extract peak-layer mean-pooled rep over
  `mrm8488/ImageNet1K-val` for a given model/kind; (b) logistic-regression probe with the fixed
  split; save `imagenet_probe_results.json`.
- Create `mono_analysis/` (mirror `neo_analysis/`/`sail_analysis/`): vendored InternVLChatModel load
  (force eager + dense mask; runtime `img_context_token_id`; likely **neo venv** due to the
  transformers-5.10 `KeyError 'type'`), `mono_cka_extract.py` (per-layer image-token reps),
  `mono_imagenet_extract.py`. Smoke first (n=8): print N_layers, img token id, n_vis, eager mask OK.
- Create `devtools/fig_internal_encoder.py` — predictivity R²(depth) curves (native rising vs
  encoder-VLM flat) + ImageNet probe-accuracy bars (native peak vs encoder ceiling vs floor).

Phases:
- **P2a (fast, do first):** re-extract reps at N=1000 for the existing models (NEO/SAIL/Gemma/LLaVA/
  Qwen + DINO/CLIP/SigLIP); compute predictivity + extend CKA. Produce the discriminative curve
  (native rising vs encoder-VLM flat). This alone is a strong result.
- **P2b:** Mono-InternVL custom dir + extraction (CKA + predictivity).
- **P2c (heavy):** download `mrm8488/ImageNet1K-val`; extract peak-layer reps per model; logistic
  probe; assemble the ceiling/floor comparison.

## 6. Data
- Predictivity / CKA images: 1000 stratified VMCBench dev (already the CKA convention; row-aligned
  across models via fixed indices).
- ImageNet: `mrm8488/ImageNet1K-val` (HF, not gated, 39.3k, `image`+`label`). Download via
  `HFOFF=0` on a compute node into `hf_cache`. Watch disk/inode (5.2 GB).

## 7. Risks & mitigations
| Risk | Mitigation |
|---|---|
| Predictivity R² high for everyone (n<H overfit) | held-out CV R² + optional PCA(X)→256; report floor (layer-0) must be low |
| Mono-InternVL load `KeyError 'type'` on transformers 5.10 | use neo venv (4.57) in the custom dir, like NEO/SAIL |
| ImageNet probe acc not comparable across sources | one fixed protocol (split/head/preprocess); report encoder ceiling + raw-pixel floor as anchors |
| Encoder-VLM control NOT flat (would break the story) | that itself is a finding — report honestly; it would mean encoder-VLMs also "rebuild" vision internally |
| CKA-to-DINO confound | predictivity + ImageNet are primary; CKA supporting only |

## 8. Deliverables
- `predictivity_results.json` + `fig_internal_encoder.png` (R²-depth curves: native rising vs
  encoder-VLM flat vs floor).
- `imagenet_probe_results.json` (native peak vs encoder ceiling vs floor top-1 table).
- Extended CKA incl. Mono-InternVL.
- A `neo_report/INTERNAL_ENCODER.md` write-up: does the evidence support "native grows an internal
  vision encoder"? — with the discriminative controls front and center, and honest caveats.
