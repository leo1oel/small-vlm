# P1 — New-model architecture generalization: Implementation Plan (post-maintenance)

> **For agentic workers:** runs after the June 16–23 cluster maintenance. Builds on the validated
> P0 metric. REQUIRED SUB-SKILL: superpowers:executing-plans (or subagent-driven-development).
> Steps use checkbox (`- [ ]`) syntax.

**Goal:** Test whether the mid-stack-fusion conclusion generalizes across MoE / per-layer-visual-expert /
recipe-scaled architectures, via within-family controlled contrasts, using the P0-validated
attention-knockout φ (`fusion_window.py`) + suffix-mean-ablation `sufmeanabl` (`activation_patch.py`)
+ maturation strips, on VMCBench (+ MMStar, GQA).

**Architecture:** Add a `kind` branch per native-transformers model to the three HF probes; a custom
analysis dir for Mono-InternVL; benchmark loaders for MMStar/GQA. Each model: download → n=8 smoke
(invariants) → full pipeline → analyze against the existing baselines.

**Tech Stack:** transformers 5.10.2 (main `.venv`, classes verified present 2026-06-16); neo venv for
Mono-InternVL. SLURM `ckpt-all`/`cse-ckpt`, `HF_HOME=/mmfs1/gscratch/krishna/leoym/hf_cache`.

---

## Prerequisite P1.0 — downloads (BLOCKER; needs internet)

The login node had no internet on 2026-06-16 and these are NOT cached. Fetch into
`/mmfs1/gscratch/krishna/leoym/hf_cache` when connectivity returns (watch the inode/EDQUOT quota —
it bit us before; the 30B/26B MoEs are large).

- [ ] **Models:** `OpenGVLab/InternVL3_5-8B-HF`, `OpenGVLab/InternVL3_5-30B-A3B-HF`,
  `google/gemma-4-26B-A4B-it`, `lmms-lab/LLaVA-OneVision-1.5-4B-Instruct`,
  `lmms-lab/LLaVA-OneVision-1.5-8B-Instruct`, `lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct`,
  `OpenGVLab/Mono-InternVL-2B`, `deepseek-community/Janus-Pro-7B`.
  Command: `HF_HUB_OFFLINE=0 huggingface-cli download <id>` (or `snapshot_download`).
- [ ] **Datasets:** MMStar (`Lin-Chen/MMStar`) and GQA (`lmms-lab/GQA` testdev_balanced, or HF `gqa`).
  Verify the field names (MMStar: `question`/`answer`/options or `image`+letter; GQA: `question`,
  `answer`, `imageId`→image) before wiring the loader.
- [ ] Sanity: `ls hf_cache/hub | grep -iE 'internvl3_5|gemma-4-26|onevision-1.5|onevision-2|mono-internvl|janus'`.

---

## Task 1: `internvl` kind branch (dense↔MoE pair) — Easy

InternVL3.5-8B-HF (dense) and -30B-A3B-HF (MoE) share `InternVLForConditionalGeneration` (verified
present) and `image_token_id=151671`; plain causal mask, no DeepStack. ONE branch serves both.

**Files:** Modify `devtools/fusion_window.py`, `devtools/pathway_maturation.py`,
`devtools/activation_patch.py`, `devtools/freeze_probe.py` (add `kind=="internvl"`).

- [ ] **Step 1:** In each probe's loader `if/elif` chain add:
```python
elif kind == "internvl":
    from transformers import InternVLForConditionalGeneration as M
```
The shared `_build` (apply_chat_template + `proc(images=[image], text=[prompt])`) and
`img_id = config.image_token_id` (151671) path already cover it — no special-casing expected.
- [ ] **Step 2 (smoke, n=8):** `python devtools/activation_patch.py OpenGVLab/InternVL3_5-8B-HF internvl /tmp/s.jsonl 8 8 sufmeanabl`
  Verify: prints `img_id=151671`, `n_vis>0`, no `skip`, INV `sufmeanabl[0]==intact`. ⚠️ InternVL is
  native-resolution → variable n_vis (denoise will skip; sufmeanabl is self-aligned, fine). Confirm
  intact acc is in InternVL3.5-8B's published VMCBench range.
- [ ] **Step 3:** full pipeline on BOTH dense + MoE:
```bash
for M in OpenGVLab/InternVL3_5-8B-HF OpenGVLab/InternVL3_5-30B-A3B-HF; do
  T=$(basename $M); MODEL=$M KIND=internvl OUT=neo_analysis/results_sufpatch_$T.jsonl N=1000 NC=200 MODE=sufmeanabl sbatch devtools/activation_patch.slurm
  MODEL=$M KIND=internvl OUT=neo_analysis/results_win_$T.jsonl sbatch <fusion_window.slurm> 1000 200 both
done
```
  (30B-A3B needs a100/h200 for memory; ~3B active keeps it fast.)
- [ ] **Step 4:** `python devtools/patch_analysis.py neo_analysis/results_sufpatch_<dense>.jsonl internvl <knockout>` for each; **the contrast = does the MoE q50 ≈ the dense q50?** (predict: yes — MoE only changes the FFN, not the shared attention that carries fusion).

## Task 2: `gemma4moe` kind branch (Gemma dense↔MoE) — Easy

`google/gemma-4-26B-A4B-it` (sparse MoE, `enable_moe_block:true`, id 258880). ⚠️ The existing
`gemma` kind uses `Gemma4UnifiedForConditionalGeneration`; the 26B-A4B may load with
`Gemma4ForConditionalGeneration` (both verified present). Determine the correct class on first load.

- [ ] **Step 1:** add branch; try `Gemma4ForConditionalGeneration` first, fall back to the Unified
  class. Reuse the existing `gemma` `_build` path (it uses `apply_chat_template(..., tokenize=True,
  return_dict=True)` — keep that; Gemma-4 needs the image inside the chat content). img_id=258880.
- [ ] **Step 2 (smoke n=8):** verify class loads eager, `img_id=258880`, `n_vis>0`, invariants.
- [ ] **Step 3:** full pipeline; **contrast against the EXISTING `results_*_gemma.jsonl` (12B dense)**
  — same family, dense↔MoE.

## Task 3: `janus` kind branch (unified) — Easy

`deepseek-community/Janus-Pro-7B`, `JanusForConditionalGeneration` (verified), id 100594, 30 layers.

- [ ] **Step 1:** add branch. ⚠️ Janus has decoupled understanding/generation paths — probe the
  UNDERSTANDING path: pass `generation_mode="text"` (or the processor's understanding template) and
  ensure the SigLIP→projector image tokens (not the VQ generation tokens) are the ones at `img_id`.
  Verify on smoke that `n_vis` matches the SigLIP token count, not the VQ stream.
- [ ] **Step 2-3:** smoke + full pipeline.

## Task 4: LLaVA-OneVision recipe ladder (`onevision15`, `onevision2`) — Medium

Self-contained trust_remote_code (NO external pip). `lmms-lab/LLaVA-OneVision-1.5-{4B,8B}-Instruct`
(`LLaVAOneVision1_5_text` backbone) and `lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct`
(`llava_onevision2`), all id 151655, Qwen3-style.

- [ ] **Step 1:** add a TRC loader branch: `AutoModelForImageTextToText.from_pretrained(model_id,
  trust_remote_code=True, attn_implementation="eager")` (or the bundled class). ⚠️ **Source-verify on
  first smoke that eager materializes a 4D mask** (the bundled modeling has its own
  `_update_causal_mask`); if it forces sdpa/flash, patch like SAIL. anyres → variable n_vis (sufmeanabl
  self-aligned; denoise skips).
- [ ] **Step 2-3:** smoke each; full pipeline on 4B + 8B (size scaling) + OV-2 (newest recipe). The
  **recipe ladder** = existing OneVision-7B (2024) → OV-1.5-4B/8B → OV-2: does fusion depth drift?

## Task 5: Mono-InternVL-2B custom dir (Claim B) — Medium

`OpenGVLab/Mono-InternVL-2B`, older `internvl_chat` AutoModel+TRC, per-layer visual-FFN expert
(= this repo's `feat/visual-ffn-expert` arch). Mirror `neo_analysis/`/`sail_analysis/`.

- [ ] **Step 1:** create `mono_analysis/` with: vendored/loaded modeling (force eager + dense 4D
  mask), `mono_common.py` (build_inputs→(b,vm), `img_context_token_id` resolved at runtime),
  `mono_fusion_full.py` (knockout φ), `mono_activation_patch.py` (sufmeanabl, copy the SAIL hook
  pattern), `mono_pathway_mat.py`. Run in neo venv (or main if internvl_chat loads there).
- [ ] **Step 2-3:** smoke (verify `<IMG_CONTEXT>` id, n_vis>0, eager 4D mask materializes — the key
  risk); full pipeline. **Claim B test:** is the image-stream maturation distributed across ALL
  layers (per-layer visual expert) rather than front-loaded? Does fusion still sit mid-stack?

## Task 6: Benchmarks — MMStar (MCQ) + GQA (generative)

- [ ] **MMStar:** add a `load_mmstar()` + `doc_to_prompt_mmstar()` to the probes' dataset interface
  (MCQ, reuse letter scoring; MMStar is curated vision-indispensable → larger R₀). Run φ + sufmeanabl
  on a diagnostic subset: {LLaVA-1.5, Qwen2.5-VL, NEO, + one new MoE, + Chameleon if added}.
- [ ] **GQA:** add `load_gqa()` + generative scoring = NLL of the gold short answer's token(s) with
  the donor-swap R₀ (no letter logits). Add a `score_generative(logits_seq, answer_ids)` helper.
  Run on the same diagnostic subset. **Tests whether fusion depth is benchmark- and task-invariant.**

## Task 7: Analysis, figures, write-up

- [ ] Extend `patch_analysis.py` to print the dense↔MoE contrast table and the size/recipe ladder.
- [ ] Update `fig_patch_triangulation.py` MODELS list with the new anchors; regenerate.
- [ ] Add the new models to `fig_dol_pretty.py` (division-of-labor) + `fig_two_conclusions.py`.
- [ ] Write `neo_report/GENERALIZATION_RESULTS.md`: the dense↔MoE verdict (does MoE move fusion?),
  the size/recipe verdict (scale-migration claim), the Claim-B verdict (Mono-InternVL), the
  benchmark-invariance table. Update `CROSS_VALIDATION.md` §4.

## Optional follow-on (if native results are ambiguous)

- [ ] Add `chameleon` + `emu3` kind branches (classes verified present; native-VQ, id from
  vocabulary_map) and run the **direct Narrow-Gate (2412.06646) replication** — does our φ/sufmeanabl
  show their early [EOI] gate or mid-stack fusion? This is the decisive native-architecture test.

## Per-model validation invariants (gate every model before trusting its numbers)
Print `N_layers / img_id / n_vis / letter_ids`; `assert n_vis>0`; `R₀>0.05`; intact acc within the
model's published VMCBench range; `sufmeanabl[0]≡intact` exact; (φ) `suf[0]≡cost[N-1]`. For native-res
models confirm `denoise_skip=nvis_mismatch` is expected and sufmeanabl still runs.

## Self-review notes
- All P1 native classes verified present in transformers 5.10.2 (2026-06-16); the ONLY blind risks
  are per-model chat-template/img-id/eager-mask details (Tasks 2-5) — each gated by an n=8 smoke.
- Datasets + models are uncached with no login-node internet → P1.0 download is a hard prerequisite.
- sufmeanabl (P0-validated) is the primary new metric; denoise/prefix-meanabl are non-discriminative
  (do not use). See `neo_report/CROSS_VALIDATION.md` §3.
