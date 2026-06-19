# Native-VLM ablation — running results (auto-accumulated while user sleeps)

*Branch `feat/visual-ffn-expert`. Spec `docs/superpowers/specs/2026-06-14-native-vlm-capacity-readiness-design.md`.
All arms: same bee-mix blend / 5000 steps / seed 42 / token budget → directly comparable. Single seed.*

## Benchmark scores (lmms-eval, checkpoint-5000)

| arm | lever | vmc_avg | vmc_reason | vmc_doc | vmc_ocr | vmc_general | POPE | MMVP |
|---|---|---|---|---|---|---|---|---|
| baseline | — | 0.396 | 0.407 | 0.304 | 0.400 | 0.451 | 0.565 | 0.510 |
| capacity (visual-FFN expert) | ① in-stack capacity | 0.403 | 0.433 | 0.320 | 0.410 | 0.434 | 0.598 | 0.537 |
| supervision (aim_pixel) | ② pixel supervision | 0.393 | 0.417 | 0.308 | 0.360 | 0.443 | 0.637 | 0.510 |
| nepa-only | ② feature supervision | — running (36136662) | | | | | | |
| capacity+nepa | ①+② | — running (36136225) | | | | | | |
| prefix (K=6) | ③ early/input capacity | — pending (36136794) | | | | | | |

**Scores read:** capacity helps modestly but on all three (POPE +3.3, MMVP +2.7, vmc +0.7, mostly reason/doc/ocr);
pixel-supervision helps POPE a lot (+7.2) but flat on vmc/mmvp. Both real, both partial, different axes.

## ⚠️ FDI / readiness mechanism (aux_fusion_full, VMCBench dev-1000, 250 causal) — the important result

| arm | intact | swap | **R0 (usable image signal)** | L_f50 | functional CoM | rho@L1 |
|---|---|---|---|---|---|---|
| baseline | 0.395 | 0.370 | **+0.025** | 8 | 15.9 | 0.717 |
| capacity | 0.387 | 0.367 | **+0.020** | 2 | 14.5 | 0.586 |
| aim_pixel | 0.393 | 0.368 | **+0.025** | 1 | 11.4 | 0.651 |
| **nepa (feature sup)** | 0.392 | 0.358 | **+0.034** ← only arm to RAISE R0 | 2 | 13.5 | — |
| capacity+nepa (①+②) | 0.372 | 0.368 | **+0.004** ← LOWEST (destructive) | — | — | — |

**UPDATE (capacity+nepa FDI, 2026-06-18):** combining the two levers **kills** image-
conditioning: capacity alone 0.020, nepa alone 0.034, but **together 0.004** (near zero).
The visual-FFN expert (extra capacity) gives the model more ways to route around using
the image; stacking nepa on top does not rescue it — the objectives interact
destructively for R0. Bee-mix R0 ranking: nepa 0.034 > baseline/aimpixel 0.025 >
capacity 0.020 > capacity+nepa 0.004. **All ≤0.034 — architecture/supervision levers on
bee-mix barely move R0.** Contrast the real levers: E4 vision-pathway (encoder 0.197 vs
pixel 0.059, 3.3×) and data blend (energon-vision 0.059 > bee-mix 0.025). The grounding
loss (direct R0 optimizer, launched 36169764) is the targeted attempt to move it.

**UPDATE (nepa FDI):** nepa (feature-level supervision) is the **only** lever that raised R0 (0.025→0.034, ~+36%
rel; swap acc 0.370→0.358 = leans more on the *right* image). capacity LOWERED it (0.020), pixel-sup was flat
(0.025). This matches the spec's prediction that **feature** supervision beats pixel/capacity for image-readiness,
and refines the headline: among the three levers, the one touching the binding problem (image-conditioning) is the
**from-data feature supervision**, not capacity. Still tiny absolute (NEO 0.31) and single-seed — confirm with the
nepa benchmark scores (eval running) and capacity+nepa (does adding capacity on top help or not).

**HEADLINE (honest):** R0 ≈ **0.02** for every arm — swapping in a WRONG image barely changes the answer, so the
native 2B answers VMCBench mostly from **language priors, not the image**. Reference: NEO-2B on this exact probe has
intact 0.71 / R0 **0.31** (~15× more image-dependent). Our model is both lower-accuracy AND barely image-conditioned.

Consequences:
1. **Neither capacity nor pixel-supervision raised R0** → they don't fix the binding problem (image-conditioning).
2. The apparent FDI shift (L_f50 8→1/2, CoM down) is over a **near-noise ±0.02 signal** → NOT interpretable; the
   spec's "capacity raises readiness / moves FDI earlier" hypothesis is **not supported** here.
3. Strong support for the original **data/under-training caveat**: NEO ~385M pairs vs our ~3.5M; R0≈0.02 looks like a
   model that never learned to use the image. Architecture levers can't rescue a model below its image-use floor.

**Implication for the program:** the highest-value next move is likely the deferred **data/steps/resolution** levers
(does R0 rise with more data/steps?), and checking image-conditioning directly — not more capacity/supervision arms.
Still finish nepa / capacity+nepa / prefix (esp. prefix = the on-thesis "build vision before read"); but set
expectations: if R0 stays ~0.02, none of them is the fix.

Probe files: `neo_analysis/results_fusion_full_{baseline,capacity,aimpixel}5000.jsonl`.

## ⭐ E2 — R0-over-training dynamics (2026-06-18, the "stuck basin" evidence)

baseline (native PIXEL, bee-mix) R0 by training step (FDI probe, VMCBench dev-1000):

| step | intact | swap | **R0** |
|---|---|---|---|
| 500 | 0.349 | 0.343 | +0.006 |
| 1000 | 0.335 | 0.319 | +0.016 |
| 2000 | 0.381 | 0.357 | +0.024 |
| 3500 | 0.393 | 0.368 | +0.025 |
| 5000 | 0.395 | 0.370 | **+0.025** |

**R0 plateaus at ~0.025 by step 2000 and never climbs.** This is the "stuck in the
language-prior basin" signature: if the native model were merely under-trained, R0 would
keep rising with steps — instead it SATURATES early at a tiny value, i.e. the model gives
up on developing image-use by step 2000. Supports the training-dynamics / shortcut
hypothesis (H_shortcut) over the under-training hypothesis (H_slow). The accuracy gain
splits ~half prior (swap 0.343→0.370) / ~half image-use (R0 0.006→0.025), both saturating.
Decisive complement (running): the ENCODER trajectory at matched early steps — if CLIP R0
is already high at step 1000, the data/steps are sufficient to learn image-use and pixel's
failure is the PATHWAY (shortcut), not data. Probes: `results_fusion_full_baseline{500..3500}.jsonl`.

## ⭐ E4 — encoder vs pixel, MATCHED (2026-06-18, the headline causal control)

Same LM (Qwen3-1.7B), same data (energon-vision), same schedule, step 20000 —
**the only difference is the vision pathway** (CLIP-B/16 encoder vs raw patches).
`aux_fusion_full.py` extended to probe the encoder arm (encode_images branch).

| arm | n | intact | swap | **R0 (image signal)** |
|---|---|---|---|---|
| **ENCODER** (sft-clip) | 1000 | 0.596 | 0.399 | **+0.197** |
| **PIXEL** (sft-unified) | 1000 | 0.468 | 0.409 | **+0.059** |

**Encoder R0 / pixel R0 = 3.3×.** Two decisive reads:
1. **swap acc is identical** (0.399 vs 0.409) → both fall into the SAME language-prior
   basin; with a wrong image they answer from priors at the same rate.
2. **The entire intact-accuracy gap is image-use.** Encoder's +0.128 intact edge ≈
   its +0.138 R0 edge. The encoder didn't get a better prior — it learned to CONDITION
   on the image 3.3× more, and that conditioning IS the accuracy difference.

→ Answers the core question ("why does encoder look at the image but raw-pixel
doesn't"): matched everything else, the vision pathway is the lever, and it acts by
raising R0 (image-conditioning), not by improving the prior. Strong support for the
training-dynamics / shortcut hypothesis: the encoder makes "use the image" a reachable
descent direction; raw pixels leave it expensive so the model stays in the prior basin.
(Note: pixel R0 0.059 here > bee-mix baseline 0.025 → the data blend itself moves R0,
energon-vision native conditions more than bee-mix. Encoder 0.197 still < NEO-2B 0.31,
a far larger run. Single-seed.) Probes: `results_fusion_full_e4_{clip,pixel}20k.jsonl`.
