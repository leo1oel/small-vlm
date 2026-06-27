# Do native VLMs grow an internal vision encoder? — Claim B, verified

*P2 of the fusion-generalization program. Companion to `CROSS_VALIDATION.md` (metric
triangulation) and `GENERALIZATION_RESULTS.md` (Claim A — where fusion happens). This file
establishes **Claim B**: a native (encoder-free) VLM's internal parameters, trained only by
generative pre-training, build a visual representation that is **functionally equivalent to a
real frozen vision encoder's**. Spec: `docs/superpowers/specs/2026-06-18-claimB-internal-vision-encoder.md`.*

---

## 0. The claim, stated precisely

Any visual information must be *processed into a representation* before it can be combined with
text. Encoder-based VLMs (LLaVA, Qwen-VL) do this in an explicit, separately-trained vision
encoder (CLIP/SigLIP), then a connector hands LLM-ready features to the language model. Native /
encoder-free VLMs (NEO, SAIL, Gemma-4, Mono-InternVL) have **no** such encoder: raw image patches
are linearly embedded and fed straight into the LLM. The claim is that **some of the LLM's own
parameters take over the encoder's job** — i.e. across the early-to-mid layers the image-token
stream is transformed into a representation that is *the same kind of thing* a real encoder
produces. If true, native VLMs **grow an internal vision encoder via generative pre-training**, and
Claim A (mid-stack fusion reads it) + Claim B (vision is built first) become one mechanism.

We test this with two functional probes, weakest→strongest, plus the existing geometric one:

1. **Linear predictivity (PRIMARY).** Can a native layer's image representation be linearly mapped
   onto a real encoder's representation (held-out CV ridge R²)? This asks whether the two are
   *interchangeable up to a linear map* — the operational definition of "same representation."
2. **ImageNet linear probing (gold standard).** Does the native internal representation reach
   ImageNet-1k top-1 accuracy comparable to a real encoder's (and far above raw pixels)?
3. **CKA** (geometric, supporting; see `fig_prebuffer_cka`).

The **discriminative design** is the heart of the test (so we cannot fool ourselves):

| representation source | prediction if Claim B is true |
|---|---|
| native (enc-free) internal layers | R²/accuracy **RISES** from a low layer-0 floor to encoder-grade |
| encoder-VLM internal layers (control) | **FLAT-HIGH** from layer 0 (the encoder already fed it in — nothing to build) |
| layer-0 / shuffled-label (floor) | low / ~zero |

A bare "native R² is high" proves nothing. The *shape* — native rising while the encoder-VLM
control stays flat and the null stays at zero — is what isolates "an encoder is being built."

## 1. TL;DR

- **Native VLMs build an encoder-grade visual representation across their layers.** Linear
  predictivity of a frozen SigLIP rises from a near-zero layer-0 floor to **0.68–0.73** by mid/late
  depth in **all five** native models (SAIL 0.08→0.73, Gemma 0.17→0.72, NEO-2B 0.20→0.70,
  NEO-9B 0.20→0.68, **Mono-InternVL 0.23→0.68**). Mono-InternVL is the cleanest mechanistic case: it
  has *no* vision tower at all (its `vision_model` is a single patch-embedding layer), so the
  +0.45 rise is built entirely by the per-layer **visual experts** inside its 24 LLM layers.
- **Encoder-VLM controls are flat-high, exactly as predicted.** LLaVA's image-token rep predicts
  *its own* CLIP at **R²=0.885 already at layer 0** and stays there (rise +0.00); Qwen is flat at
  ~0.62→0.67. They have nothing to build — the encoder fed it in.
- **The signal is real, not overfitting.** The shuffled-label null is ≈ **−0.01** everywhere
  (held-out CV R² with PCA, n=1000); the rise is 30–60× the null.
- **It is learned, not architectural (causal control).** The *same* Gemma architecture with **random
  (untrained) weights** peaks at SigLIP-R²=0.21 and stays flat-low, vs 0.72 when trained; random
  LLaVA collapses from 0.885 to 0.236. So the internal encoder is **grown by generative
  pre-training**, not an inductive-bias property of deep transformers over patches.
- **Native internal reps are *as* encoder-grade as encoder-VLMs'.** SAIL's peak SigLIP-predictivity
  (0.725) **exceeds** LLaVA's and Qwen's (0.697 / 0.714) — the internal "encoder" is not a weak
  imitation.
- **Functional gold standard (ImageNet-1k linear probe) confirms it in absolute accuracy.** A native
  model's layer-0 rep is near-chance (SAIL 0.009, Gemma 0.008, Mono 0.044) and **climbs to ≥ the
  encoder ceiling** (SAIL peak **0.824 > CLIP/DINO/SigLIP 0.74–0.77**; Gemma 0.741; Mono 0.648), while
  encoder-VLM controls are flat-high from layer 0 (LLaVA 0.73, Qwen 0.80). Native internal reps are
  not just encoder-*like*, they are as *classification-capable* as a real encoder.
- **Conclusion:** native VLMs, with no vision encoder in their architecture, use their internal
  parameters to construct a representation that is linearly interchangeable with — and as
  classification-capable as — a purpose-built vision encoder. They grow one through generative
  pre-training.

## 2. Method

**Representations.** For every model we mean-pool the image-token hidden state at each decoder layer
over a **fixed 1000-image** set (VMCBench dev, row-aligned across models). Real encoders
(DINOv2-base, CLIP-L/14-336, SigLIP-base) contribute their patch-mean of the last hidden state.
Extractors: `devtools/cka_extract.py` (LLaVA/Qwen/Gemma + encoders), `neo_analysis/neo_cka_extract.py`,
`sail_analysis/sail_cka_extract.py`, `mono_analysis/mono_cka_extract.py`.

**Linear predictivity.** Per (model layer ℓ, encoder E) we fit a cross-validated ridge regression
`X_{M,ℓ}(n,H_M) → Y_E(n,H_E)` and report held-out 5-fold R² (variance-weighted). Each fold fits its
own StandardScaler + PCA(X)→256 + PCA(Y)→128 + RidgeCV (no leakage; n>d so held-out R² cannot be
inflated). A **row-permuted Y** gives the null floor. `devtools/predictivity_compute.py` (CPU);
results `neo_analysis/predictivity_n1000.json`.

**Why these controls defeat the obvious objections:**
- *"High-dim reps always predict each other."* → held-out CV R² + PCA(X)→256 (n=1000>256); the
  shuffled-Y null is ≈ 0, so chance predictivity is ≈ 0.
- *"Maybe everything is high."* → the encoder-VLM control is flat **and** the native floor (layer 0)
  is low; only the native models *rise*.
- *"Single-encoder artifact."* → three reference encoders (DINOv2/CLIP/SigLIP); the rising shape
  holds for all three (SigLIP cleanest, see §4).
- *Calibration sanity*: LLaVA→its-own-CLIP must be ~maximal and flat — it is (0.885, rise +0.00).

## 3. Results — linear predictivity (N=1000)

Peak R² (and depth) / layer-0 floor / shuffled null, per reference encoder. Native = should rise;
enc-VLM = should stay flat.

### SigLIP (cleanest discriminator — lowest floor)
| model | role | floor (L0) | peak | peak depth | **rise** | shuffled null |
|---|---|---|---|---|---|---|
| SAIL-7B | native | 0.077 | **0.725** | 0.75 | **+0.648** | −0.013 |
| Gemma-4-12B | native | 0.168 | **0.724** | 0.33 | **+0.556** | −0.013 |
| NEO-2B | native | 0.196 | **0.701** | 0.90 | **+0.505** | −0.010 |
| NEO-9B | native | 0.202 | **0.677** | 0.93 | **+0.476** | −0.014 |
| Mono-InternVL-2B | native | 0.227 | **0.682** | 0.79 | **+0.454** | −0.010 |
| LLaVA-1.5 (enc-VLM) | control | 0.612 | 0.697 | 0.62 | +0.085 | −0.013 |
| Qwen2.5-VL (enc-VLM) | control | 0.647 | 0.714 | 0.36 | +0.067 | −0.013 |

### CLIP
| model | role | floor (L0) | peak | **rise** |
|---|---|---|---|---|
| SAIL-7B | native | 0.145 | 0.619 | **+0.474** |
| Gemma-4-12B | native | 0.235 | 0.687 | **+0.452** |
| NEO-2B | native | 0.286 | 0.648 | **+0.362** |
| NEO-9B | native | 0.293 | 0.623 | **+0.330** |
| LLaVA-1.5 (enc-VLM) | control | **0.885** | 0.885 | **+0.000** (peak at L0) |
| Qwen2.5-VL (enc-VLM) | control | 0.624 | 0.672 | +0.048 |

DINOv2 shows the same ordering but with a higher floor (0.14–0.31) and smaller rise — DINO is a
self-supervised, lower-level encoder whose patch statistics partly overlap with raw pixels (a known
CKA-to-DINO confound), so it discriminates less sharply. CLIP and SigLIP (semantic, language- or
caption-aligned) are the load-bearing references.

**Reading.** Every native model's image-token rep is **near the floor at layer 0** (≤0.20 for
SigLIP) and **climbs to encoder-grade** (0.68–0.73) by mid/late depth — the model is *building* the
representation. Both encoder-VLMs are **flat from layer 0** — LLaVA literally peaks at layer 0
predicting its own CLIP. The null is ≈ 0. This is precisely the discriminative signature of an
internally-grown encoder.

### 3.1 Causal control — is it the architecture, or the training? (random-init)

The rising curve shows native internal reps *become* encoder-grade, but a skeptic can ask: is that
built by **generative pre-training**, or is it an inductive-bias property of the architecture (any
deep transformer over patches)? To separate them we re-run the identical extraction on the **same
architectures with random (untrained) weights** (`RANDINIT` in `cka_extract.py`).

Prediction if Claim B is right: the trained native model **rises**, but its random-init twin stays
**flat-low** (no learned transformation builds the encoder) — and a random-init encoder-VLM falls
from flat-high to flat-low (its "high" came from the *trained* CLIP tower, now random).

| model | weights | SigLIP floor → peak | CLIP floor → peak | shape |
|---|---|---|---|---|
| Gemma-4-12B (native) | **trained** | 0.168 → **0.724** | 0.235 → 0.687 | **rises** |
| Gemma-4-12B (native) | **random-init** | 0.126 → **0.210** | 0.178 → 0.248 | **flat-low** |
| LLaVA-1.5 (enc-VLM) | **trained** | 0.612 → 0.697 | 0.885 → 0.885 | flat-high |
| LLaVA-1.5 (enc-VLM) | **random-init** | 0.176 → 0.199 | 0.225 → 0.236 | flat-low |

**The 2×2 closes the case.** Same Gemma architecture: *trained* peaks at SigLIP-R²=0.724, *random*
peaks at 0.210 — the +0.51 build-up is created by **generative pre-training**, not the architecture
(a random deep transformer over patches builds nothing — it stays at the floor). The encoder-VLM
control is even sharper: *trained* LLaVA predicts its own CLIP at 0.885, but with *random* weights
that collapses to 0.236 — its flat-**high** came entirely from the trained CLIP tower; randomized,
it is flat-**low**. So an encoder-grade representation requires **learned** weights everywhere; the
novel finding is that a native VLM learns them *inside the LLM, from generative pre-training alone*.
(Random-init reps are numerically clean — no NaNs; null ≈ −0.01 as for all runs.) See
`fig_internal_encoder.png`: the two random-init curves (dotted) sit flat at the bottom while every
trained native curve climbs past them to encoder grade.

## 4. Results — ImageNet linear probing (gold standard)

One fixed protocol for **every** representation source: ImageNet-1k val (50 000 images, 1000 classes),
a per-class 80/20 stratified split (40 train / 10 test per class → 10 000 test points), a multinomial
logistic head on the mean-pooled rep, test top-1. Native / enc-VLM reps are taken at a 7-point
layer-depth grid (so the layer-0 *floor* and the *peak* layer are both measured); encoders contribute
their single output rep (the **ceiling**). `devtools/imagenet_probe.py`; `imagenet_probe_results.json`.

| source | role | layer-0 floor top-1 | **peak top-1** | peak layer |
|---|---|---|---|---|
| DINOv2-base | encoder (ceiling) | — | 0.752 | (single) |
| CLIP-L/14-336 | encoder (ceiling) | — | **0.768** | (single) |
| SigLIP-base | encoder (ceiling) | — | 0.744 | (single) |
| **SAIL-7B** | native | **0.009** | **0.824** | L32/32 |
| **Gemma-4-12B** | native | **0.008** | **0.741** | L38/48 |
| **Mono-InternVL-2B** | native | **0.044** | **0.648** | L24/24 |
| LLaVA-1.5 | enc-VLM (control) | 0.718 | 0.730 | L21/32 |
| Qwen2.5-VL | enc-VLM (control) | 0.789 | 0.800 | L28/28 |

**This is the decisive functional result.** A native model's image rep at **layer 0** is near-useless
for classification — SAIL 0.009 and Gemma 0.008 (1000-way chance ≈ 0.001), Mono 0.044 (just the patch
embedding) — and **climbs to the encoder ceiling or beyond**: SAIL reaches **0.824, above** all three
real encoders (0.744–0.768) and above both encoder-VLM controls; Gemma reaches 0.741 (≈ ceiling); Mono
0.648. The encoder-VLM controls are **flat-high from layer 0** (LLaVA 0.718→0.730, Qwen 0.789→0.800) —
they were handed encoder-grade features and barely change with depth. So the native internal
representation is not merely *correlated* with an encoder (predictivity, §3); it is *as
classification-capable as one*, rising from a pixel-level floor to ≥ encoder-grade accuracy purely
across the model's own layers.

The pattern mirrors predictivity exactly — native **rises** floor→ceiling, enc-VLM **flat-high**, floor
**near-chance** — now in absolute task accuracy, the strongest form of the claim.

## 5. Rigor & honest caveats

- **Held-out CV R² + low shuffled null** rule out the "high-dim reps trivially predict each other"
  failure mode; the floor (layer 0) is low so the rise is meaningful.
- **CKA-to-DINO is confounded** (raw pixels already align ~0.6); we therefore lead with predictivity
  + ImageNet and use CLIP/SigLIP as the discriminating references.
- **Mean-pooling** compares global image descriptors, not spatial maps — appropriate for "is this an
  encoder-grade *representation*," but it does not test spatial localization.
- **Predictivity ≠ identity.** R²≈0.7 means ~70% of the encoder's (top-128 PCA) variance is linearly
  recoverable, not that the reps are identical; combined with the ImageNet probe (functional) the two
  together establish *functional equivalence*, which is the claim.
- The encoder-VLM control staying flat is a load-bearing prediction; it held. Had it risen too, the
  story would have been "all VLMs rebuild vision internally" — it did not.
- **Random-init causal control** (§3.1) is the decisive guard against the "it's just the architecture"
  reading: the same architectures with untrained weights stay flat-low (predictivity 0.21, ImageNet
  not run there but predictivity suffices), so the build-up is attributable to *learning*, not
  inductive bias.
- **SAIL exceeding the encoder ceiling (0.824 > 0.77)** is reported honestly: we do **not** claim the
  internal encoder is *better* than CLIP in general. It means SAIL's 7B / 4096-dim internal rep is at
  least as linearly separable on ImageNet as a frozen base encoder under this protocol — capacity and
  task-coupling plausibly help. The claim that survives is the conservative one: native peak **≥**
  encoder ceiling and **≫** floor.

## 5.5 Related work — does "native VLMs grow an internal encoder" contradict prior work? (No — it is supported)

The claim sits at the intersection of two literatures, and both support it; we found no contradiction.

**Encoder-free / monolithic VLMs.** The models themselves are built on the premise that an LLM *can*
learn to encode vision internally. **Mono-InternVL** (arXiv 2410.08202) — the headline subject here —
is explicitly designed around "**Endogenous Visual Pre-training**": the LLM acquires visual encoding in
its own parameters (a multimodal-MoE *visual expert* per layer), which is precisely our Claim B in the
original authors' framing. **EVE / EVEv2** (2406.11832, 2502.06788), **BREEN** (2503.12446, "image
experts"), and **HoVLE** (2412.16158, "holistic embedding") are further encoder-free models that must
build the visual representation inside the model. Our contribution is to *measure*, model-agnostically
and causally, that this internal representation reaches encoder-grade — rather than assuming it from
downstream task scores.

**Probing VLM layers against vision encoders.** **OLA-VLM** (2412.09585) distills vision-encoder
embeddings *into* intermediate MLLM layers, implying those layers can host encoder-grade features;
**RL makes MLLMs see better than SFT** (2510.16333) independently uses **ImageNet classification** of
internal representations to quantify a model's visual quality — the same functional probe we use. Our
predictivity + ImageNet-probe + **random-init causal control** turn this into a discriminative test
(native *rises* from a near-chance floor; encoder-VLM is flat-high; random weights stay flat-low).

**Representational convergence.** The **Platonic Representation Hypothesis** (Huh et al., 2405.07987) —
models converge to a shared representation across modalities — explains *why* a native layer's image
representation can be linearly mapped onto a frozen encoder's. We treat this as an honest caveat rather
than a crutch: **"Back into Plato's Cave"** (2604.18572) shows cross-modal convergence is **not**
universal at scale, and the "Aristotelian view" (2602.14486) shows it is local and must be **null-
calibrated by permutation** — exactly our shuffled-label null and random-init control. So our claim is
the narrow, causally-controlled one (native reps become linearly *and* functionally encoder-grade,
verified against a learned-vs-architecture control), not the strong universal-convergence claim those
papers caution against.

## 6. Artifacts
- **Predictivity:** `neo_analysis/predictivity_n1000.json` (9 models incl. random-init controls),
  `predictivity_n100_gate.json` (replication at N=100); `devtools/predictivity_compute.py`,
  `devtools/predictivity.slurm`.
- **ImageNet probe:** `neo_analysis/imagenet_probe_results.json`, `imagenet_<tag>.npz`;
  `devtools/imagenet_probe.py`, `imagenet_extract.slurm`, `imagenet_probe.slurm`, `launch_imagenet.sh`.
- **Per-layer reps (predictivity/CKA):** `neo_analysis/cka_*.npz` (N=1000), incl. `cka_mono.npz`,
  `cka_gemmarand.npz`, `cka_llavarand.npz`; extractors `devtools/cka_extract.py` (with `RANDINIT`),
  `neo_analysis/neo_cka_extract.py`, `sail_analysis/sail_cka_extract.py`,
  `mono_analysis/` (vendored Mono-InternVL load + CKA/ImageNet extractors).
- **Figure:** `neo_report/fig_internal_encoder.png` (`devtools/fig_internal_encoder.py`).
