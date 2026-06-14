# Native (encoder-free) VLM: readiness is the causal axis, fusion depth is not

*Design doc / research program. v2 — 2026-06-14. Supersedes the v1 "capacity +
readiness program" framing. Builds on `neo_report/ENCODER_FREE_FUSION_REPORT.md`
and `docs/early-fusion-results-summary.md`, and anchors on four reference papers
(§1.4). Outside-paper numbers are from an automated literature survey and MUST be
re-verified against the primary source before any writeup; "our probes" numbers
are our own measurements.*

> **导读 (TL;DR).** 我们要提升弱的 native(encoder-free)VLM(raw patch→投影→
> Qwen3-1.7B,无 ViT,VMCBench ~0.40)。**方法借鉴已被四篇论文反复验证的"模态专属
> 视觉容量"(hard-routed visual-FFN/attn expert,从 LLM FFN 初始化,语言路径早期冻结
> delta-tuning),但这不是我们的贡献。** 我们的贡献是一个只有我们能做的**因果机制
> 论断**:
>
> **"Encoder-free VLM 弱,不是因为融合晚,而是因为视觉表征在被读取的那一层还没建好。
> 专属视觉容量之所以有效,是因为它抬高了 *readiness*,从而把因果读取点(FDI)*作为
> 结果* 往前推——而 readiness(不是融合深度)才是因果轴:我们对两者各自独立干预,
> 只有 readiness 动了 accuracy。"**
>
> 我们独有这个论断,因为我们同时握着解离的两半:**force-fusion 臂(已做,移动读取点
> →失败)** 和 **capacity 臂(借鉴方法,抬 readiness→有效,FDI 作为结果前移)**,外加
> 因果探针(FDI / CKA-to-DINO readiness / pathway maturation)。

---

## 1. Context

### 1.1 The model under study (`*-unified`, encoder-free)
- Backbone **Qwen3-1.7B** (also 0.6B), decoder-only. D=2048, 28 layers, FFN 6144
  (SwiGLU), GQA 16q/8kv, head_dim 128.
- Vision path = **a projection, not a transformer** (`connectors.py::_RawPatchEmbedder`):
  raw RGB patches (16px×3 pooling = 48px model patch) → LN → Linear(patch_dim→2048)
  → LN → +factorized XY posemb → LN → RMSNorm(no-scale) → Linear(2048→2048). ~19.5M
  params. Spatial info only in the connector posemb; LLM uses 1D RoPE.
- ≤280 image soft tokens/image. Direct SFT (LM+connector unfrozen) on a "bee-mix"
  stream (~20k steps), or staged: connector-only pretrain on ~1M Bee captions
  (`pretrain-unified-bee-48g`, 1900 steps) → joint SFT (`sft-unified-staged`, 5000 steps).
  Compute: 48G L40/L40S, ZeRO-2 bf16.
- Infra in place: aux losses `visual_aux.objective ∈ {none, aim_pixel, nepa}` ×
  `trainer.visual_aux_weight` (`modeling_vlm.py`); cross-modal masks
  (`sdpa_xmodal.py`, `xmodal_mask.py`: prefix_lm/img2q_window, per-layer override).

### 1.2 Our own measured findings (established priors)
1. Natives are architecturally early-fusion but **functionally mid-fusion**: FDI
   (causal depth where the answer reads the image) ≈ 0.48–0.59 for all natives
   (NEO/Gemma/SAIL), same as encoder-based Qwen2.5-VL (0.54). Only LLaVA-1.5 fuses
   **early** (0.33) — its CLIP encoder is contrastively **text-aligned / LLM-ready**.
2. Controlling variable = **how LLM-ready the representation is at the read point**,
   not encoder presence. Gradient: text-aligned encoder → early; generic encoder /
   native-with-pre-encoding → mid; pure raw-pixel native (SAIL) → latest.
3. **Forcing the read point does not help** (five arms windowearly/sandwich/prefixlm/
   auxexit/randpos; windowearly didn't win, randpos hurt, only sandwich nudged
   POPE/MMVP +5/+4). → **moving fusion depth is not the lever.** *(Caveat 2.4.2.)*
4. Our native 2B is weak (~0.40 VMCBench-dev avg).
5. NEO's explicit pre-Buffer (12/40 for 2B) measurably builds DINO-grade features
   early; Gemma front-loads via a heavy patch-embedder; **SAIL (raw pixel→1 linear)
   does NOT front-load, fuses latest, weakest signal.** Our model ≈ "SAIL plus a
   slightly bigger connector."

### 1.3 Decisions taken
- **Goal = BOTH** a falsifiable mechanistic story (paper) AND benchmark gains.
- **Native purity = internal modality-specific params allowed** (visual experts /
  MoT / dedicated layers), decoder-only, **no ViT at inference**. We **lean away
  from distilling a human encoder** — it re-imports the bias we want to *grow from
  data*. Distillation is an off-thesis, flagged, optional contrast arm only.
- **Data is judged NOT the current binding constraint** (user call) → data-scaling
  is demoted from a gate to a **covariate arm** (2.4.2, §5).
- **Contribution headline = causal readiness (spine A)**; from-data-vs-distillation
  (B) and the readiness-gap law (C) are supporting sub-studies.

### 1.4 Reference work — the four anchor papers and their convergent verdict
| Paper | What | Take-away for us |
|---|---|---|
| **Mono-InternVL-1.5** (2507.12566, OpenGVLab) | Native. Modality-routed **visual experts (FFN + attn Q/K/V)**, init from LLM FFN, **language path frozen** during visual pretrain (delta-tuning), staged unfreeze (EViP++). ~1.8B base + ~1.2B visual experts. | The closest analog. Defines the **method** we borrow: dedicated visual capacity + frozen-language stability. |
| **BREEN** (2503.12446) | Native, Qwen2.5-7B. **Per-layer image-expert FFN (MoT-style)** + CLIP-distilled learnable queries. ~13M pairs (≈1% of Mono). | **Killer ablation: the image-expert FFN carries almost all the gain (+7); CLIP-distillation adds only ~+0.9.** ⇒ **capacity ≫ supervision**, and capacity is **data-efficient**. |
| **Beyond Language Modeling** (2603.03276, FAIR/NYU; Tong, Fan, …, Xie, LeCun, Zettlemoyer) | Controlled from-scratch unified-VLM sweep (Transfusion-style). | **modality-FFN > shared; per-modality MoE > MoT > dense; vision is intrinsically data-hungrier than language and MoE halves that gap.** Depth-specialization *emerges* (text early, vision late) — our FDI phenomenon seen via routing. |
| **EMO** (2605.06663) | Text-only LLM MoE; experts specialize via document-boundary pooling. | Tangential. Transferable seed: image tokens = a natural "pool" → modality-specialized experts can emerge from the modality boundary alone. |

**Convergent verdict (all four + our v1 survey):** the validated lever is
**dedicated modality-specific visual capacity inside the decoder** (hard-routed by
modality, init-from-LLM, language frozen early). MoE ≥ MoT ≥ dense; **capacity ≫
distillation** (BREEN); capacity is **also a data-efficiency lever** (BREEN ≈ Mono
at 1% data; Paper-4 MoE halves vision's data-hunger exponent) — which is exactly
why we can deprioritize data-scaling without much risk.

---

## 2. The reframe, the diagnosis, and the central claim

### 2.1 Reframed axis
Drop "native vs encoder" and "early vs mid fusion." The real question:

> **Does the visual representation get enough DEDICATED CAPACITY and the right
> FROM-DATA SUPERVISION to become LLM-ready — before the layer where the answer
> reads it?**

FDI is the *symptom*; readiness is the *cause*; capacity + supervision is the lever.

### 2.2 Diagnosis (the ~0.40 floor)
A single linear projection drops image tokens into a **shared dense 1.7B** trained
with **LM cross-entropy only**: the same weights must be *both* encoder and
reasoner, fighting over one FFN/attention, with **no objective rewarding visual
quality**. = SAIL failure mode (bottom of our ranking). Net: the visual
representation never becomes LLM-ready → weak. Paper-4's "model spontaneously puts
text early, vision late" and EVEv2's "shared training shifts the LLM weights" are
the same pathology seen elsewhere.

### 2.3 The central claim (our contribution — the causal dissociation)
We can run **two independent interventions** that no prior paper holds both of:
- **Intervene on the READ POINT** (force-fusion masks: windowearly/sandwich — *done*).
  Result: **no win.** Moving *where* the answer reads doesn't help.
- **Intervene on READINESS** (add dedicated visual capacity — the borrowed expert
  method). Prediction: **win**, *and* FDI moves earlier **as a consequence** (CKA
  readiness rises first, the read-point follows).

⇒ **Readiness is the causal axis; fusion depth is epiphenomenal.** This is the
headline. It reinterprets our own falsified force-fusion result from "a dead end"
into "the negative half of a causal dissociation," and it is something Mono/BREEN/
Paper-4 cannot claim — they have capacity (correlational) but no force-fusion
control and no causal FDI/CKA probe.

### 2.4 Caveats baked in
1. **ZeRO-2 does not shard parameters** (only optimizer states + grads). Doubling
   stored params replicates full bf16 params on every GPU. Small variants (FFN-only
   early layers, image-only prefix) fit 48G; **full all-layer DaC (~2.8–3.1B) likely
   needs ZeRO-3.**
2. **Under-training / data confound.** Our force-fusion arms may have been measured
   pre-convergence; the "fusion depth is not the lever" negative must be
   **re-measured at convergence** for the dissociation to hold. And Paper-4 is real
   evidence native vision is data-hungry — so we keep **one cheap data-slope
   covariate** (§5) to defend the capacity gain against an under-data confound. Not
   a gate (user judged data non-binding); a covariate.
3. **BREEN warns supervision is small** (+0.9 from CLIP-distillation vs +7 from the
   expert). Sub-study B's effect may be modest — that's an acceptable, clean result,
   not a failure; the capacity arm is the main dish.

---

## 3. Method (borrowed & validated) — the visual-capacity instrument

This is the **instrument**, not the contribution. Implement the convergent design:

**M1 — Hard-routed visual-FFN expert (primary arm).**
- `FFN_v = deepcopy(FFN_t)`, trainable; route image (patch + connector) tokens →
  `FFN_v`, text → `FFN_t`; **deterministic by token-modality mask** (no learned
  gate, no load-balance loss). Reuse the existing image/text token mask.
- **Frozen-language delta-tuning:** freeze `FFN_t` (and the rest of the LM) during
  visual pretrain; unfreeze in staged SFT (Mono EViP++ schedule). Directly fixes the
  §2.2 "LLM cannibalized" pathology; also makes the pretrain stage *cheaper* (Adam
  state only for `FFN_v` + connector — lighter than current full SFT).
- *Cost:* +1.0B stored / ~1.7B active. Fits 48G (ZeRO-2 caveat 2.4.1).
- *Optional M1+:* add **visual attention experts** (per-modality Q/K/V), Mono-1.5
  style — more capacity, more cost; ablate separately.

**M2 — Capacity PLACEMENT (this is where our probes earn their keep — see §4).**
Two ways to spend a fixed capacity budget; our FDI/CKA probes decide which:
- *In-stack experts* (M1, all or early-K layers) — BREEN/Mono/Paper-4 default.
- *Image-only prefix* (K NEO-style image-only layers before the shared stack;
  image tokens attend bidirectionally among themselves via `sdpa_xmodal`; ~+300M for
  K=6). Concentrates capacity at the input boundary.
- **The placement question** ("where does dedicated capacity most raise readiness at
  the read depth?") is answerable *only causally*, with our probes — a genuine
  novelty over the four papers, which place capacity uniformly or let it emerge.

**M3 — Architecture family.** Default to **hard-routed modality experts (M1)** — the
cheapest validated step. **Per-modality MoE** (Paper-4: MoE > MoT) is a later
extension if M1 wins and budget allows ZeRO-3; **EMO's modality-pool emergence** is
a stretch idea for that MoE stage. Not first.

---

## 4. The contribution (spine A) + sub-studies B, C

### Spine A — causal readiness (the headline; §2.3)
Establish the dissociation with our probes:
- **A-neg (have it, re-run at convergence):** force-fusion (windowearly/sandwich) —
  moving the read point → no accuracy gain.
- **A-pos:** M1 visual-FFN expert — accuracy gain, **and** (i) CKA-to-DINO readiness
  rises at early/mid depth, (ii) FDI moves earlier, (iii) the gain tracks the
  readiness rise, not a forced read-point. Pre-registered: readiness rises *before*
  FDI shifts (cause precedes effect across the depth axis).
- **A-place:** M1-alllayers vs M1-early-K vs image-only-prefix — does placing
  capacity where the probe says the rep is under-built beat uniform placement?

### Sub-study B — from-data supervision vs encoder distillation (advances the native thesis)
Clean **2×2: {capacity M1 on/off} × {supervision off / from-data / CLIP-distill}**.
- *from-data* = existing `nepa` (self-supervised next-patch embedding) and/or
  **align image tokens to the model's OWN text embeddings** (AlignVLM-style — the
  data-grown analog of CLIP's text-alignment, no encoder imported).
- *CLIP-distill* = the off-thesis BREEN-style cosine loss to a frozen CLIP (contrast
  only, λ→0 annealed so inference stays encoder-free).
- **Question:** can from-data supervision recover BREEN's distillation gain *without*
  importing an encoder? Measure CKA-to-DINO to confirm the signal actually raises
  readiness. Expect a modest effect (BREEN caveat 2.4.3) — a clean dissociation either
  way.

### Sub-study C — the readiness-gap law (mechanistic unification)
Across *every* arm (capacity, supervision, data-slope, encoder-import contrast),
regress accuracy on **readiness = CKA-at-read-depth**. If one metric linearly
predicts accuracy across heterogeneous interventions, that's a mechanistic law none
of the four papers offer: *"close the readiness gap by any means → accuracy follows."*
Falsifier: if arms with equal readiness have different accuracy, readiness is not
sufficient and the law is wrong (still informative).

---

## 5. Experiment matrix / ablation ladder (the paper's central figure)

| Rung / arm | Config | Primary measure |
|---|---|---|
| R0 | linear connector (current, ≈SAIL) | baseline FDI / CKA / VMCBench ~0.40 |
| **A-neg** | force-fusion (windowearly/sandwich), **re-run at convergence** | read-point moved, accuracy flat → negative control |
| **A-pos** | + M1 visual-FFN expert (frozen-LM delta → staged unfreeze) | accuracy↑, CKA↑, FDI earlier — *jointly* |
| A-place | M1-all vs M1-earlyK vs image-only-prefix (param-matched) | which placement raises read-depth readiness most |
| M1+ | + visual attention experts | does attn capacity add over FFN-only |
| B-2×2 | {M1}×{none / nepa+own-text-align / CLIP-distill} | from-data vs distillation readiness recovery |
| COV-data | baseline + M1 at 1M/5M/15M | covariate: defend gain vs under-data confound |
| C-regress | all arms | accuracy ~ readiness(CKA@read-depth) |

**Pre-registered predictions:** A-pos wins and readiness rises *before* FDI shifts;
A-neg stays flat; in-stack-early ≈ or > prefix for read-depth readiness (test
Paper-4's "vision-late" against NEO's "front-load"); B's from-data recovers *most*
of the small distillation gain; C is a tight positive regression.

**The story:** *"A weak native VLM is weak because its visual representation is
under-built at the read point, not because it fuses late. We dissociate the two —
forcing the read point fails (negative control), while dedicated visual capacity
raises readiness and the read-point follows as an effect — establishing readiness,
measured by CKA/FDI, as the causal axis. A from-data readiness signal recovers most
of what encoder distillation buys, so the encoder can stay grown-from-data."*

---

## 6. Measurement plan
Reuse probes: FDI / retained(d) / ρ (`*_fusion_full.py`), pathway maturation
(`pathway_maturation.py`), CKA-to-DINO readiness (`cka_extract.py`+`cka_compute.py`),
lmms-eval (VMCBench-dev / POPE / MMVP / OCRBench). **At every arm report (accuracy,
FDI, CKA-readiness) together** — the joint move *is* the claim. Single seed for the
sweep; 2-seed confirm on A-pos and A-neg (the dissociation hinges on them).

## 7. Recommended sequencing
1. **A-pos + A-neg-rerun** first — the dissociation is the headline and reuses
   existing force-fusion arms + the borrowed M1 method (staged recipe, lower VRAM in
   the frozen stage). This alone is a paper-shaped result.
2. **A-place** + **B-2×2** next (placement + the native-thesis ablation).
3. **COV-data** in parallel as cheap insurance; **C-regress** is computed over all
   arms at the end (no new training).
4. Hold MoE (M3), visual-attention experts (M1+), and CLIP-distill (B contrast) as
   extensions.

## 8. Risks / open questions
- **Supervision effect may be small** (BREEN +0.9) — B is a clean dissociation, not
  a guaranteed win; don't oversell.
- **Data-hunger (Paper-4)** could confound the capacity gain — COV-data defends it;
  if the capacity gain vanishes at low data, that's itself a finding.
- **Under-training confound** invalidates A-neg unless re-run at convergence — do it.
- **Warm-started experts may not diverge** from text FFN at our data scale → check
  expert-divergence (weight/representation drift) as a sanity probe.
- **ZeRO-3** needed for MoE / full-DaC variants.
- **Literature numbers unverified** (Mono +114, BREEN +7/+0.9, Paper-4 exponents,
  MoE>MoT) — verify primary sources before citing.

## 9. Success criteria
- **The dissociation holds:** A-pos beats baseline with the (accuracy↑, CKA↑, FDI
  earlier) joint signature; A-neg (at convergence) stays flat. → readiness causal,
  fusion-depth epiphenomenal.
- **At least one capacity arm beats baseline-mix on VMCBench** (benchmark goal).
- **B gives a clean read** on from-data vs distillation (recovers most / doesn't).
- **C** is reproducible as the unifying figure (accuracy ~ readiness across arms).
