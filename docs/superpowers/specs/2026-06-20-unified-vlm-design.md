# Unified Understanding + Generation VLM (encoder-free, Qwen3 + visual-FFN expert)

- **Date:** 2026-06-20
- **Status:** Design (pre-implementation)
- **Branch context:** builds on `feat/visual-ffn-expert`
- **Reference:** MiniT2I / MM-JiT — https://github.com/Hope7Happiness/minit2i-torch (blog: https://peppaking8.github.io/#/post/minit2i). Kaiming He's group; pixel-space flow-matching T2I.

## 1. Goal

A single model that does **both** image understanding (image→text) **and** image generation (text→image), built on our existing encoder-free stack (Qwen3 backbone + raw-patch connector + per-layer modality-routed visual-FFN expert). Generation uses **continuous flow matching in pixel/patch space** (Transfusion-style), reusing the visual-FFN expert as the modality-specific pathway.

**Ambition: capability model** — chase GenEval / DPG-Bench on generation while preserving understanding benchmarks (lmms-eval / MME …). Not a toy demo.

**Success criteria**
- Generation: competitive GenEval / DPG-Bench at the chosen resolution (reference: MiniT2I-B/16 = 0.873 GenEval).
- Understanding: no large regression vs the current understanding-only VLM on MME etc.
- Causal claim: the **visual-FFN expert** (arm A) measurably helps the unified model vs a **fully-shared dense backbone** (arm C, control).

## 2. Background

### 2.1 What we already have (reuse)
- **Qwen3-1.7B** decoder, plain pre-norm (RMSNorm), 1D RoPE, 28 layers, D=2048, SwiGLU FFN 6144.
- **`_RawPatchEmbedder`** (`connectors.py`): raw RGB patches → LN → Linear → +factorized 2D-XY posemb → RMSNorm → Linear. Variable-resolution, ≤280 patches.
- **Visual-FFN expert** (`modeling_vlm.py`: `install_visual_experts`, `_routed_mlp_forward`, `_set_visual_mask`): per-layer sibling `mlp.mlp_visual`, hard per-token modality routing `out = text·(1−m) + visual·m`, init-copied from text FFN.
- **Cross-modal mask infra** including `cross_modal_mask_mode=prefix_lm` (configs `sft-unified-bee-mix-prefixlm.yaml` already exist).
- **`visual_aux_head`** (next-patch / aim_pixel / nepa): a head that already outputs in patch space — seed for the x-prediction head.
- **Energon streaming** from Azure blob (`energon_dataset.py`): `{folder: weight}` + per-folder jsonl + lazy media.

### 2.2 What MiniT2I / MM-JiT actually is (verified from source)
- Pixel-space **flow matching**, **x-prediction with v-loss** (direct ε-/v-prediction collapse in high-dim pixel space; `diffusion.py:30-34`).
- **No adaLN, no timestep conditioning at all** — `model.py:342-343` computes `vec = t_embed(t)+pooled_embed(pooled)` then `del vec`; blocks take no conditioning. "Noise level remains visible from the corrupted image input." Plain pre-norm transformer. Ablation: removing adaLN and reallocating params to depth *improved* FID (18.7→13.7).
- **Double-stream** MM-DiT block: separate `norm/qkv/proj/mlp` per modality; joint (fully bidirectional, `is_causal=False`) attention over `[txt; img]`; 2D-RoPE on image, 1D on text.
- **Input not normalized**: `BottleneckPatchEmbed` = Conv(3→128, k=p,s=p) → Conv(128→hidden,1×1), no LayerNorm — preserves noise magnitude.
- Text: frozen **T5** (1024-d) + 2 text-adapter blocks; CFG via `label_drop_rate=0.1` replacing dropped text with a learned `mask_token`.
- Recipe: `t ~ sigmoid(N(−0.8, 0.8))`, `noise_scale=2.0`, `x_t = x₁·t + ε·(1−t)` (t=1 clean), zero-init final layer, 100-step Euler, CFG≈6.
- Resolution: main models 512²/16px = **1024 image tokens** (256-token variant is 512²/32px). Compute: B/32 ablation ≈ 3 days × 8×H100.

## 3. Follow vs. Adapt (do NOT copy blindly)

MiniT2I is a **from-scratch, dedicated** T2I DiT. We graft generation onto a **pretrained causal LLM**. Copy what is representation-level robust; adapt what depends on from-scratch dynamics, the LLM being causal/pretrained, or the LLM's normalization.

| Aspect | MiniT2I | Our choice | Why |
|---|---|---|---|
| Per-block adaLN modulation | yes (then removed in MM-JiT) | **removed** | Would be surgery on Qwen3's norms; MM-JiT shows it's unnecessary. ✓ follow |
| Timestep signal | **none** (del vec) | **keep, as an in-context `t` token** (LLM-native, no surgery); ablate removal | Qwen3 RMSNorm washes magnitude and was never trained to read noise from input; "no t at all" is risky on a pretrained backbone. **Adapt** |
| Output parametrization | x-pred + v-loss | **x-pred + v-loss** | JiT result is representation-level; transfers. ✓ follow |
| Image RoPE | 2D RoPE | **keep Qwen3 1D RoPE + connector additive 2D posemb** | Swapping RoPE disrupts pretrained attention. **Adapt** (2D-RoPE a later ablation) |
| Modality separation | double-stream (attn+FFN) | **FFN-only expert** (shared attention QKV) + control C (fully shared) | Reuse existing expert; Transfusion shows shared attn works. Decided. **Adapt** |
| Text encoder | frozen T5 + adapters | **native Qwen3 in-context**, prefix-LM | The LLM *is* the text model; unification's whole point. **Adapt** (risk §8.1) |
| Input embed | conv, no norm | **separate generation input branch that preserves scale** (bypass `patch_ln1`) | Our connector's leading LN kills noise scale. **Adapt** |
| noise_scale / t-dist | 2.0 / (−0.8,0.8) | **starting points, retune** | Effective SNR differs after the connector projection. **Adapt** |
| Init / LR | from-scratch, single LR | **pretrained backbone + expert init-from-text; differential LR; zero-init head** | Cold-start a foreign objective on a pretrained LM without wrecking it. **Adapt** |
| Sampler | 100-step Euler (+MeanFlow distill) | same; distillation deferred | transfers. ✓ follow |

## 4. Architecture

Components (NEW = to add):

| Component | Source | Role |
|---|---|---|
| Qwen3 decoder | existing | shared backbone, **plain pre-norm, zero surgery** |
| `_RawPatchEmbedder` (understanding) | existing | clean patches → tokens (keeps input LN) |
| **Generation input branch** | NEW | noised patches → tokens, **scale-preserving** (no leading LN) |
| 2-way FFN expert (`mlp`/`mlp_visual`) | existing | text FFN / visual FFN; image tokens (clean & noised) → visual FFN |
| **Timestep token** | NEW | one in-context token from `TimestepEmbedder(t)`, prepended to the image block (no adaLN) |
| **x-prediction head** | NEW (extend `visual_aux_head`) | hidden → patch (predicted clean `x̂₀`), **zero-init** |
| `lm_head` | existing | text logits (understanding) |
| Control arm C | config flag | `visual_expert=false` → single dense FFN does both |

### 4.1 Data-flow modes
```
Understanding (img→text), EXISTING:
  image → _RawPatchEmbedder(clean) → [img tokens] ⊕ text
  → Qwen3 (causal) ; img tokens → visual FFN (no t token)
  → lm_head → next-token CE   (loss: text only)

Generation (text→img), NEW:
  text (prefix) ⊕ [t token] ⊕ [N noised image slots @ t,  x_t=x₁·t+ε·(1−t)]
  noised patches → generation input branch (scale-preserving) → img tokens
  → Qwen3 ; prefix-LM mask (text bidirectional prefix; image block bidirectional, attends text+t)
  img tokens → visual FFN
  → x-pred head → x̂₀ ; v-loss   (loss: image only)
```

## 5. Training

### 5.1 Objectives (joint, from Qwen3 base)
- **Understanding microbatch:** next-token CE on text; image = clean patches; causal mask. (unchanged)
- **Generation microbatch:** flow-matching v-loss. Sample `t ~ sigmoid(N(μ,σ))`; `ε = randn·noise_scale`; `x_t = x₁·t + ε·(1−t)`; model outputs `x̂₀`; `loss = MSE((x̂₀−x_t)/(1−t)_clamp, (x₁−x_t)/(1−t)_clamp)`; loss on image tokens only.
- **CFG dropout:** drop text on `label_drop_rate≈0.1` of generation samples → unconditional (replace text prefix with a learned null embedding).
- **Joint:** per-microbatch separation (a microbatch is all-understanding or all-generation), sampled by ratio `p_gen`, gradient-accumulated. Loss weight `λ` balances CE vs v-loss (different scales).

### 5.2 Data
- **GPIC** (Azure `data/gpic/test/`, ~1M, `jsonl_name: test.jsonl`, account `vigstandard`). Records are **already messages format** (`{messages:[{role:user, content:[{type:image, path:"images/<id>.jpg"}]}, {role:assistant, content:"<caption>"}], width, height, caption_type}`).
  - **Understanding direction:** consume messages as-is (existing energon path). Zero new data code.
  - **Generation direction:** reverse — caption (`messages[-1].content`) = prompt; image (`messages[0]...path`) = target; resize to canvas, emit target patches + noised input. NEW transform.
- **Understanding blend:** existing Bee + OV1.5 + FineVision (optional, to keep understanding strong).
- **Knobs:** `caption_type` (short/medium/long) weighting toward denser captions for compositional generation; `p_gen`; per-source blend weights.

### 5.3 Attention / position
- Understanding: existing causal (+ optional cross-modal windows).
- Generation: `cross_modal_mask_mode=prefix_lm` — text prefix bidirectional, image block fully bidirectional, image attends text+t; **text does not attend the noised image** (so text KV is reusable across all sampler steps — see §6).
- Image spatial info: connector additive factorized 2D posemb + Qwen3 1D RoPE. (2D RoPE = later ablation.)

### 5.4 Optimization (adapted for pretrained backbone)
- Init: Qwen3 pretrained; visual expert init-from-text; **x-pred head + generation input branch + timestep token zero/near-zero init**.
- **Differential LR:** lower for pretrained backbone, higher for NEW generation params. Warmup + cosine.
- `noise_scale`, `t`-distribution `(μ,σ)`, `λ`, `p_gen` are **tunables** (start from MiniT2I values, retune).
- ZeRO-2 bf16, chunked CE (existing). Gradient checkpointing.

## 6. Inference
- Understanding: AR decode (unchanged).
- Generation: encode text prefix once (cache KV) → init image slots `randn·noise_scale` → 100-step Euler `t:0→1`: per step `x̂₀ = model(...)`, CFG `x̂₀ = uncond + (cond−uncond)·scale` (scale≈6), `v=(x̂₀−x_t)/(1−t)_clamp`, `x += v·Δt` → unpatchify → `clamp(−1,1)`.
- Resolution: choose canvas per sample; **start 256-token (e.g. 256²/16px or 512²/32px)** for cheap iteration, scale to 512²/16px (1024 tokens). Aspect buckets allowed (GPIC native ~640 long side → target ≤512²).
- MeanFlow 4-step distillation: deferred.

## 7. Evaluation
- Generation: **GenEval + DPG-Bench** (reuse MiniT2I's `evaluation/geneval` + `scripts/evaluate_dpg_bench.py`); FID optional.
- Understanding: lmms-eval (MME …), existing.
- **Control comparison:** arm A (FFN expert) vs arm C (fully shared) under identical data/compute — the causal claim for the expert.

## 8. Risks & open questions
1. **Causal-LM text features may under-condition T2I** (SD3/MiniT2I use bidirectional T5 for a reason). Mitigation: prefix-LM bidirectional text prefix; optional lightweight text-adapter blocks; fallback to richer conditioning if GenEval stalls. **Highest risk.**
2. **Read/write interference in the shared visual FFN** (arm A). Diagnostics: generation v-loss plateau or understanding regression → upgrade to 3-way expert or double-stream.
3. **Timestep handling:** verify the in-context `t` token suffices; ablate against (a) adding `t` to image embeddings, (b) pure MM-JiT no-`t`. Confirm the scale-preserving input branch actually exposes noise level.
4. **Pretrained-causal → bidirectional-denoising shift** is large; joint-from-base may be slow to co-adapt. Watch early dynamics; zero-init head limits cold-start damage.
5. **Recipe transfer:** noise_scale / t-distribution tuned for from-scratch raw-pixel DiT; retune for our connector SNR.
6. **Compute realism:** MiniT2I B/32 ablation ≈ 3 days × 8×H100; on L40s expect longer. Use the 256-token config for bring-up; budget the 1024-token capability run explicitly.
7. **Data scale:** GPIC 1M is a solid bring-up but ~1/12 of CC12M; top GenEval may need more data / higher-quality dense captions.

## 9. Phasing
1. **Bring-up (gen-only sanity):** generation path at 256 tokens on GPIC; overfit a small set; confirm sampler produces coherent images. Validates head/loss/mask/sampler + scale-preserving input + `t` token.
2. **Joint A:** understanding + generation joint-from-base, 256 tokens; tune `p_gen`, `λ`, LRs.
3. **Control C:** same as (2) with `visual_expert=false`.
4. **Scale:** 512²/16px (1024 tokens), longer schedule, GenEval/DPG; add dense-caption finetune if needed.

## 10. File-level change plan (initial)
- `connectors.py`: add scale-preserving generation input branch.
- `modeling_vlm.py`: timestep token injection; x-pred head (extend `visual_aux_head`); generation forward path + prefix-LM wiring; reuse `_set_visual_mask` routing for noised image tokens.
- new `diffusion.py` (under `src/vlm/`): flow-matching loss (x-pred/v-loss), `t` sampler, Euler sampler, CFG.
- `energon_dataset.py`: generation sample transform (reverse direction from messages; resize-to-canvas; target patches + noised input).
- `config_schema.py` + new config(s): generation config block (resolution, patch, noise_scale, t-dist, p_gen, λ, label_drop, cfg), GPIC dataset config (`folders: {gpic/test:1.0}`, `jsonl_name: test.jsonl`).
- `train.py` / `set_trainable.py`: objective mixing (per-microbatch), differential LR groups (gen params).
- eval: import MiniT2I geneval + dpg scripts.

## 11. Out of scope (YAGNI for v1)
- MeanFlow / few-step distillation.
- 3-way / double-stream experts (only if arm A shows interference).
- 2D-RoPE on image tokens.
- Image editing / interleaved generation / multi-image.
- T5 or any external text encoder.
