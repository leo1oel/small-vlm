# Where and how do vision and text fuse — encoder-free vs encoder-based VLMs

*VMCBench dev (1,000 multiple-choice items). All numbers are measured, not
estimated; every claim links to the probe that produced it. Generated overnight
2026-06-09; all 10 models and all four metric families complete.*

> **⚠️ v2 修正(2026-06-09 晨,见 `FUSION_WINDOW_REANALYSIS.md`)**:本报告的*测量数字*
> 全部经独立复算成立,但两处*解释*已被后续实验修正:
> 1. **"融合深度由视觉特征的 LLM-就绪度/文本对齐度决定"已被推翻**(Qwen2.5-VL 的 ViT 同样
>    经 CLIP 对比预训练却中层融合;CKA 成熟度与融合深度不相关)。正确框架:**融合深度 =
>    训练把 image→question 读取电路布线在哪**;中层是大规模训练的普遍吸引子(文献共识),
>    LLaVA-1.5 的"早"是其极简配方的特例。
> 2. FDI 是质心。v2 补测了 **suffix-blocking(真实融合起点)**与 **lastrow 分解**
>    (证明可用图像信号走 image→question-tokens→answer,答案位直读不承重),
>    中层结论在"起点"口径下依然成立。

---

## 0. TL;DR — the answer to your question

You asked: *is encoder-free fusion later because the model must spend early
layers encoding the image (doing the vision encoder's job), or is something
else going on?*

The evidence says **both halves of the dichotomy are partly wrong, and the
truth is sharper**:

1. **Some encoder-free models spend their early layers encoding the image —
   but it is an architectural choice, not an inevitable cost of having no
   encoder.** NEO (which has 12 explicit "pre-Buffer" layers) and Gemma
   front-load image processing (image-update center-of-mass 0.40–0.50 of depth,
   2–3× more early image-update than LLaVA; §4.2). **But SAIL-7B — the purest
   encoder-free model (raw pixels → one linear layer → Mistral) — does NOT
   front-load** (imgCoM 0.56, like the encoder models): it processes image
   tokens distributed across depth. So "early layers replace the vision encoder"
   is real and measurable *for NEO/Gemma by design*, and false for SAIL.

2. **Functional fusion — the depth at which the answer actually starts
   *reading* the image — sits at mid-stack for every encoder-free model (FDI
   0.48–0.57 for NEO/Gemma, 0.59 for SAIL), and Qwen2.5-VL, which HAS a vision
   encoder, fuses at the same mid-depth (0.54).** The only model that fuses
   *early* is LLaVA-1.5 (0.33) — which also has an encoder. (§4.1)

3. **So fusion depth is not controlled by "encoder vs no encoder." It is
   controlled by how LLM-ready the visual features are when the text query
   reads them**, and the models fall on a clean gradient of exactly that:
   text-aligned encoder (LLaVA's CLIP) → **early 0.33**; generic encoder or
   native-with-pre-encoding (Qwen / NEO / Gemma) → **mid 0.48–0.57**; pure
   raw-pixel native with no encoder and no pre-Buffer (SAIL) → **latest 0.59**.
   The user's intuition ("no encoder → spend layers → fuse later") holds
   *cleanly only for SAIL*, the purest case; it fails for NEO/Gemma, whose
   early "encoding" runs in parallel and leaves fusion at the same mid-depth as
   an actual encoder model (Qwen).

So encoder-free models are **architecturally early-fusion** (image and text
tokens are mixed from layer 0) but **functionally mid-fusion**, and the
"encoding tax" they pay runs *in parallel* with text processing in the early
stack rather than serially delaying fusion. The single thing that separates
them from the best encoder model (LLaVA) is not the encoder's existence but the
encoder's **text-alignment / LLM-readiness**.

---

## 1. Models and why each is classified as it is

Classification verified from configs/weights, not assumed.

**Encoder-free / "native" (raw patches → linear projection → token stream
inside the LLM; no vision transformer):**

| Model | LLM backbone | Decoder layers | image tokens are… |
|---|---|---|---|
| NEO1.0-2B (MT, SFT) | Qwen3-1.7B | 40 (0–11 "pre-Buffer") | 14px patches → linear |
| NEO1.0-9B (MT, SFT) | Qwen3-8B | 42 (0–5 "pre-Buffer") | 14px patches → linear |
| NEO1.5-2B (SFT) | Qwen3-1.7B | 40 | 14px patches → linear |
| NEO1.5-9B (SFT) | Qwen3-8B | 42 | 14px patches → linear |
| Gemma-4-12B-it | Gemma-3/4 text stack | 48 | linear patch embedder |
| SAIL-7B | Mistral-7B | 32 | 14px patches → **one** linear layer |

- *Gemma-4-12B is genuinely encoder-free*: its `vision_config`
  (`model_type: gemma4_unified_vision`) has **no `num_hidden_layers` and no
  attention heads** — it is a patch embedder (`patch_size`, `mm_embed_dim`,
  pooling, soft-tokens), not a transformer tower. Confirmed from the cached
  config.
- *SAIL-7B is the purest case*: `MMistralForCausalLM`, a vanilla 32-layer
  Mistral whose image tokens are raw 14×14 pixel patches passed through a single
  `vis_embed` linear layer and summed into the embedding stream (image tokens
  form a bidirectionally-attended prefix). It is also the ByteDance SAIL paper's
  own model (arXiv 2504.10462), which makes it the decisive test of that paper's
  attention claim (§4.3).
- NEO releases only SFT for the 9B/1.5 lines other than NEO1.0 (HF has
  NEO1.0-2B and 1.0-9B in PT/MT/SFT; NEO1.5 only SFT), so the MT/SFT pair is
  reported for NEO1.0 at both sizes, SFT only for NEO1.5.

**Encoder-based (vision transformer → projector → token stream):**

| Model | encoder | LLM | layers |
|---|---|---|---|
| LLaVA-1.5-7B | CLIP ViT-L/14-336 | Vicuna-7B | 32 |
| Qwen2.5-VL-7B | native-resolution ViT | Qwen2.5-7B | 28 |

These two are deliberately different: LLaVA uses a frozen **contrastively
text-aligned** CLIP encoder; Qwen2.5-VL uses its own jointly-trained
dynamic-resolution ViT. That difference turns out to matter a lot (§4.1).

---

## 2. The three axes — and why they are not the same thing

A recurring error in this literature is to read one axis as if it were another.
We measure three **orthogonal** things:

1. **Representational** — *is image information present in the hidden state?*
   (ρ, §3.2). Present ≠ used.
2. **Functional** — *is the image being used to compute the answer?* (cost /
   retained / FDI, §3.1). This is the one that matters for "fusion."
3. **Attention mass** — *how much attention probability lands on image tokens?*
   (FastV, §3.3). This is descriptive and, as we show, **misleading**: a token
   can carry an answer-determining image signal while receiving almost no
   attention, because attention sinks dominate the mass.

The headline result of the whole study is that these three axes disagree, and
only the functional axis answers "where does fusion happen."

---

## 3. Methods (precise definitions)

All scoring is **single-forward letter scoring**: we read the logits of tokens
`A/B/C/D` at the answer position (not generation). VMCBench dev is
category-ordered (20 categories × 50), so every causal sweep uses a **stratified
subset** (stride across the full 1,000) to avoid the category bias that made an
earlier "first-N" run wrong. Representational/attention metrics use all 1,000.

### 3.1 Functional fusion — cost(d), retained(d), FDI
For each decoder layer cut depth d we **block text-query→image-key attention in
layers [0..d)** (4D-mask surgery) and score the answer with (a) the real image
and (b) a swapped wrong image (donor = item (i+37) mod N).
- `R0 = intact_acc − swap_acc` = the usable image signal (how much the *right*
  image helps over a wrong one).
- `retained(d) = acc(real, block[0..d)) − acc(swap, block[0..d))`;
  `retained(d)/R0` = fraction of usable image signal still alive after blocking
  the first d layers.
- **FDI (Fusion Depth Index) = funcCoM/N**, the depth center-of-mass of the
  per-layer marginal drop of retained(d), normalized to [0,1]. FDI<0.33 early,
  0.33–0.66 mid, >0.66 late. *Probe: `*_fusion_full.py`.*

### 3.2 Representational content-fusion — ρ(l)
At the answer position, `ρ(l) = ‖h_l(I) − h_l(I')‖ / ‖h_l(I) − h_l(∅)‖`
(real vs wrong image, over real vs no image). How much the hidden state encodes
*which* image. *Same probes.*

### 3.3 Visual- vs text-pathway maturation — u_img(l), u_txt(l)
Per layer, the relative residual-stream update of each token,
`‖h_l[p] − h_{l−1}[p]‖ / ‖h_{l−1}[p]‖`, averaged separately over image tokens
and text tokens. **imgCoM** = depth center-of-mass of u_img (where the image
stream is transformed); **imgEarly%** = fraction of total image update in the
first quarter of layers. *Probe: `pathway_maturation.py`, `neo_pathway_maturation.py`,
`sail_pathway_mat.py`.*

### 3.4 Attention allocation — FastV last→image
FastV's exact definition (from its code): mean over heads → last-token query row
→ sum over image-token keys = fraction of answer-position attention on the
image; plus the share on the pre-image "sink" tokens. *Probe: `encoder_vlm_attn.py`,
`sail_attn.py`.*

### 3.5 Visual-encoding depth — CKA to a frozen encoder
Linear CKA between each model's per-layer mean-pooled image-token reps and a
**frozen reference vision encoder** (DINOv2 / CLIP-L / SigLIP) over a fixed
100-image set. Rising CKA over early layers = the decoder is *building* encoder-
grade features; high CKA at layer 0 = the image arrived already encoded.
*Probe: `cka_extract.py` + `cka_compute.py`.*

---

## 4. Results

### 4.1 Functional fusion depth (FDI) — the core result

`intact` and `swap` are letter accuracies; FDI is relative depth.

Causal sweep on a stratified subset spanning all 20 categories (n_causal in the
table); intact/swap on all 1,000.

| Model | family | N layers | n_causal | intact | swap | R0 | **FDI** |
|---|---|---|---|---|---|---|---|
| NEO1.0-2B-SFT | free | 40 | 200 | 0.71 | 0.41 | +0.31 | **0.54** |
| NEO1.0-2B-MT | free | 40 | 334 | 0.71 | 0.40 | +0.31 | **0.57** |
| NEO1.0-9B-SFT | free | 42 | 334 | 0.76 | 0.45 | +0.31 | **0.50** |
| NEO1.0-9B-MT | free | 42 | 334 | 0.75 | 0.46 | +0.29 | **0.52** |
| NEO1.5-2B-SFT | free | 40 | 334 | 0.77 | 0.40 | +0.37 | **0.54** |
| NEO1.5-9B-SFT | free | 42 | 284 | 0.77 | 0.46 | +0.31 | **0.53** |
| Gemma-4-12B | free | 48 | 200 | 0.79 | 0.39 | +0.40 | **0.48** |
| **SAIL-7B** | free | 32 | 250 | 0.73 | 0.45 | +0.28 | **0.59** |
| LLaVA-1.5-7B | enc | 32 | 200 | 0.52 | 0.31 | +0.22 | **0.33** |
| Qwen2.5-VL-7B | enc | 28 | 166 | 0.76 | 0.40 | +0.36 | **0.54** |

**Reading it:**
- **Every encoder-free configuration is functionally mid-to-late fusion, FDI
  0.48–0.59** — NEO × {1.0,1.5} × {2B,9B} × {MT,SFT} cluster at 0.50–0.57,
  Gemma at 0.48, and the purest model **SAIL at 0.59 (the latest of all)**.
  Invariant to family, scale, and training stage.
- **LLaVA fuses early (0.33)** — its CLIP features are so LLM-ready that the
  answer reads them from the bottom of the stack. (`intact=0.52` matches LLaVA's
  published VMCBench ≈51.8, validating the measurement.)
- **Qwen2.5-VL fuses mid (0.54)** — *with* an encoder, yet sitting squarely
  inside the encoder-free cluster.

➡ **Having an encoder does not, by itself, move fusion earlier.** Only a
strongly text-aligned encoder (LLaVA's CLIP) does; Qwen's encoder leaves fusion
at the same mid-depth as the encoder-free models, and the purest encoder-free
model (SAIL) fuses the latest of all. *(Figure: `fig_fdi.png`.)*

### 4.2 The visual pathway: front-loaded in NEO/Gemma, not in SAIL

| Model | family | imgCoM | imgEarly% | (where image worked on) |
|---|---|---|---|---|
| NEO1.0-2B-SFT | free | 0.49 | 0.30 | early-to-mid |
| NEO1.0-9B-SFT | free | 0.42 | 0.47 | **front-loaded** |
| NEO1.0-9B-MT | free | 0.40 | 0.51 | **front-loaded** |
| NEO1.5-2B-SFT | free | 0.50 | 0.28 | early-to-mid |
| NEO1.5-9B-SFT | free | 0.44 | 0.42 | front-loaded |
| Gemma-4-12B | free | 0.45 | 0.43 | front-loaded |
| SAIL-7B | free | **0.56** | **0.20** | **distributed (like encoders!)** |
| LLaVA-1.5-7B | enc | 0.56 | 0.15 | image barely touched early |
| Qwen2.5-VL-7B | enc | 0.53 | 0.24 | later |

**This is the place the simple hypothesis breaks — and the data is honest about
it.** The "early layers do the encoder's job" signature is **real but
architecture-specific, not a universal property of being encoder-free:**
- **NEO (all sizes) and Gemma front-load image processing** (imgCoM 0.40–0.50,
  imgEarly 0.28–0.51 — 2–3× LLaVA's 0.15). NEO does this *by design*: it has
  **12 explicit "pre-Buffer" layers** (6 for 9B) whose stated job is to encode
  the image before the LLM proper; Gemma's patch-embedder design similarly. So
  for these models the "spend early layers replacing the encoder" story is
  literally built into the architecture and is measurable.
- **SAIL does NOT front-load** (imgCoM 0.56, imgEarly 0.20) — it processes its
  image tokens distributed across depth, statistically *indistinguishable from
  the encoder-based models*. SAIL has no pre-Buffer: it drops raw patches into a
  standard Mistral with a bidirectional image prefix, and the visual refinement
  is spread through the stack.

➡ So "encoder-free ⇒ early visual encoding in the first layers" is **true for
NEO/Gemma (which have explicit pre-encoding stages) and false for SAIL.** The
front-loading is a consequence of *specific architectural choices* (a pre-Buffer
/ heavy patch embedder), not of lacking an encoder. CKA (§4.4) resolves whether
SAIL still reaches encoder-grade features, just later and distributed.

Critically, **none of this changes the functional fusion depth**: NEO, Gemma,
AND SAIL all fuse at mid-stack (§4.1) regardless of whether their visual
encoding is front-loaded or distributed. The visual-encoding work — wherever it
sits — runs *in parallel with* text processing and does not serially delay the
answer's read of the image. *(Figure: `fig_maturation.png`.)*

### 4.3 Attention mass — and the SAIL-paper reconciliation

FastV last→image fraction (mean over layers), plus the pre-image "sink" share:

| Model | family | last→image (mean) | last→image peak | sink share |
|---|---|---|---|---|
| LLaVA-1.5-7B | enc | 0.086 | 0.51 @ L0 | 0.63 |
| Qwen2.5-VL-7B | enc | 0.030 | 0.07 @ L27 | 0.41 |
| Gemma-4-12B | free | 0.188 | 0.43 @ L39 | 0.30 |
| NEO1.0-2B (POPE*) | free | 0.11–0.23 | 0.48–0.53 @ L12 | — |
| **SAIL-7B** | **free** | **0.061** | **0.35 @ L0** | **0.53** |

**The SAIL-7B result is decisive — and it contradicts the SAIL paper on the
SAIL paper's own model.** Measured with FastV's *exact* definition (mean over
heads → answer-position query → image keys) on VMCBench, **SAIL-7B routes only
6.1% of answer-position attention to its ~800–2000 image tokens** — 53% goes to
the single sink token and **94% to text overall**. That is squarely in the
*modular* range (LLaVA 8.6%, Qwen 3.0%), not the **60–80% the SAIL paper claims
for single-transformer MLLMs.**

Reconciliation (not hand-waving — from the per-layer curve):
- The high numbers only exist in the **first layer**: SAIL peaks at 35% @ L0
  and collapses to ~5% by L2 (LLaVA likewise 51% @ L0 → collapse). This is
  exactly FastV's own finding (image attention is shallow-only, which is *why*
  FastV prunes image tokens after layer 2). A 60–80% claim is only reachable by
  reporting **shallow layers** and/or **summing over the very large image-token
  count** (SAIL has up to ~2000 image tokens vs ~50 text tokens, so even
  near-uniform attention concentrates mass on image by sheer count) — not by the
  answer-position, depth-averaged FastV metric, which gives 6%.
- Encoder-free Gemma (0.19) is the only model meaningfully above the modular
  pair, but SAIL (0.06) ≈ LLaVA (0.09): **"single-transformer ⇒ 60–80% image
  attention" does not replicate under the stated metric.**
- In every model the **attention sink dominates** (LLaVA 63%, SAIL 53%, Qwen
  41%, Gemma 30%), confirming FastV's sink finding across the board.
- *NEO attention is on POPE n=30 (legacy); the VMCBench encoder-free attention
  story rests on Gemma and SAIL.*

**Crucially**, attention mass and functional fusion are *orthogonal*: Gemma puts
the most attention on the image (0.19) yet fuses no earlier (FDI 0.48) than NEO,
and LLaVA fuses earliest (0.33) on only 0.09 attention. Attention mass is **not**
a measure of fusion. *(Figure: `fig_attention.png`.)*

### 4.4 CKA — where image tokens become "encoder-grade"

Linear CKA between each model's per-layer mean-pooled image-token reps and a
**frozen neutral encoder (DINOv2** — used by none of these models). Entry =
layer-0 (post-projection, pre-decoder); we report entry CKA and the early-layer
rise. *Method validation:* against **CLIP-L**, LLaVA's image tokens peak at
**0.89 @ layer 2** — i.e. they are maximally CLIP-like right where LLaVA's CLIP
encoder feeds in, exactly as they should be. So the metric is meaningful.

| Model | family | DINO @ entry (L0) | DINO @ first-¼ | early rise | reading |
|---|---|---|---|---|---|
| NEO1.0-2B-SFT | free | 0.25 | **0.68** | +0.43 | **builds DINO-grade in pre-Buffer** |
| NEO1.0-9B-SFT | free | 0.50 | **0.73** (peak @ rel 0.12) | +0.23 | **builds early (6 pre-Buffer layers)** |
| Gemma-4-12B | free | 0.36 | **0.70** | +0.34 | **builds early, sustains** |
| SAIL-7B | free | **0.61** | 0.54 | ≈0 | **enters correlated, flat (no build)** |
| LLaVA-1.5-7B | enc | 0.33 | 0.33 | rises *late* (0.74 @ rel .75) | CLIP ≠ DINO; LLM neutralizes late |
| Qwen2.5-VL-7B | enc | **0.69** | 0.65 | ≈0 | ViT already DINO-grade at entry |

**This is the direct, converging evidence for the visual-pathway story:**
- **NEO (both sizes) and Gemma enter the stack NOT aligned to a neutral encoder
  (DINO CKA 0.25–0.50) and climb to encoder-grade alignment within the first
  quarter of layers** — for NEO this early peak coincides with its **pre-Buffer
  region** (12/40 layers for 2B → rel 0.30; 6/42 for 9B → rel 0.14, where the
  peak literally sits at rel 0.12). So NEO/Gemma's early layers *measurably do
  the vision encoder's job*, in agreement with the M2 front-loading (§4.2).
- **SAIL enters already correlated (0.61) and stays flat** — raw 14px pixel
  patches carry enough low-level structure to correlate with DINO from the
  start, so SAIL does not exhibit the "build-up" (again agreeing with M2: SAIL
  does not front-load).
- **Encoder models carry pre-encoded features**: Qwen's ViT is DINO-grade at
  entry (0.69, flat); LLaVA's CLIP is *contrastive* and not DINO-like (0.33),
  and the LLM only drifts it toward neutral-visual late.

Caveat: CKA-to-DINO is confounded as an absolute "encoder-grade" yardstick (raw
pixels already score 0.6; CLIP ≠ DINO), so we read it as a **relative trajectory**
within each model. The robust, repeatable claim is the **early build-up in
NEO/Gemma vs the flat profiles of SAIL/Qwen** — which is exactly what the
independent M2 maturation metric shows. *(Figure: `fig_cka.png`.)*

---

## 5. Synthesis — the visual pathway, the text pathway, and their interaction

Putting the axes together gives a single coherent mechanism. The controlling
variable is **how LLM-ready the visual representation is at the point where the
text query reads it**, and the models line up on a clean gradient of exactly
that quantity:

| Entry visual representation | examples | FDI |
|---|---|---|
| text-aligned encoder (CLIP) | LLaVA-1.5 | **0.33** (early) |
| generic encoder / native with a pre-encoding stage | Qwen2.5-VL, NEO, Gemma | **0.48–0.57** (mid) |
| pure raw-pixel native, no encoder, no pre-Buffer | SAIL-7B | **0.59** (latest) |

- **Text pathway:** in every model the early layers build the text query (text
  tokens carry the bulk of the early residual update). The query is not "mature"
  enough to interrogate the image until mid-stack — this is an **LLM-depth
  property** and it sets a *floor* on fusion depth (~0.5) that applies to
  everyone except the model whose image is already perfectly LLM-aligned.
- **Visual pathway:** encoder-based models receive an already-encoded image and
  mostly carry it (LLaVA/Qwen barely update image tokens early; CKA flat-high
  for Qwen). The pre-Buffer natives (NEO, Gemma) *build* encoder-grade features
  in their first layers (high early image-update, CKA climbs from 0.25→0.70 in
  the first quarter). The pure native (SAIL) drips visual processing across the
  whole stack.
- **Interaction (fusion):** the answer reads the image when **both** streams are
  ready. LLaVA's CLIP features are LLM-ready at entry → fusion can start at the
  bottom (0.33). Everyone else waits for the mid-stack text-query maturation
  floor (0.48–0.57). SAIL, whose raw-pixel visual stream is the slowest to reach
  a usable form, fuses **latest** (0.59).

**So the encoder-free models do pay an "encoding tax" — but for NEO/Gemma they
pay it in parallel in the early layers (alongside text processing), which is why
it does not push their fusion past the ~0.5 LLM floor.** The one place the user's
original intuition holds cleanly is **SAIL**: the purest encoder-free model, with
no encoder and no pre-encoding stage, both fails to front-load visual processing
*and* fuses the latest of all — there, the lack of any visual pre-processing
infrastructure does correlate with the latest fusion. But the dominant
determinant across the whole set is **visual-feature LLM-readiness on entry**,
not the binary presence/absence of an encoder (Qwen, *with* an encoder, fuses at
the same mid-depth as the pre-Buffer natives).

---

## 6. Caveats / limitations

- "Fusion" here = the causal text→image read for the multiple-choice answer on
  VMCBench; other tasks (grounding, OCR, counting) may localize differently.
- FDI causal sweeps use stratified subsets (200–334 items spanning all 20
  categories, n_causal per model in the §4.1 table); intact/swap and the
  representational/maturation metrics use all 1,000. The encoder-free FDI
  cluster has been stable across subset sizes (e.g. SAIL 0.59–0.60 from
  n_causal 119→250).
- The L1 u_img/u_txt ratio is confounded by entry-norm differences across
  architectures and is **not** used for conclusions; imgCoM / imgEarly% (shape
  of the per-model normalized update curve) are the maturation signals.
- NEO attention is on POPE (legacy n=30); Gemma + SAIL carry the VMCBench
  encoder-free attention claim.
- Only two encoder-based models; LLaVA-vs-Qwen already shows the encoder class
  is not monolithic, but more encoders would sharpen the "alignment, not
  encoder-presence" claim.

---

## 7. File index

Probes: `*_fusion_full.py` (FDI/ρ), `pathway_maturation.py` (+neo/sail) (M2),
`encoder_vlm_attn.py` / `sail_attn.py` (M4), `cka_extract.py` + `cka_compute.py`
(M5). Synthesis: `analysis_master.py` → `master_results.json`. Figures:
`encoderfree_figures.py` → `fig_fdi.png`, `fig_maturation.png`,
`fig_attention.png`, `fig_cka.png`.
