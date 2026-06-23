# Cross-validation & architecture-generalization of the VLM fusion-depth study

*Design spec, 2026-06-15. Extends the `neo_report/` study (two headline claims about
where vision–text fusion happens in VLMs). Goal: (1) cross-validate the existing
conclusions with external literature + an orthogonal causal metric, and (2) test how
far they generalize across newer architectures (MoE, native per-layer visual-expert,
new encoder-based, unified any-to-any) and a second/third benchmark.*

---

## 0. Context — what we are validating

The existing study (`neo_report/`) makes two headline claims, measured on VMCBench dev
(1,000 MCQ) with single-forward letter scoring and a causal attention-knockout metric
(φ / FDI), plus stream-freezing, suffix/lastrow knockout, CKA, and a FastV attention read:

- **Claim A — mid-stack fusion is universal.** The answer-relevant cross-modal read sits
  mid-stack (median φ 0.48–0.62) regardless of vision encoder; only LLaVA-1.5 (small-scale
  frozen-CLIP recipe) fuses early, and fusion migrates back to mid-stack as the recipe scales
  (1.5 → NeXT → OneVision). Mid-stack is framed as a universal attractor of large-scale training.
- **Claim B — vision is built before fusion.** Visual features are built unimodally before
  fusion, wherever the capacity lives (encoder / NEO pre-Buffer / interleaved early LLM layers);
  the native "encoding tax" runs in parallel with text and does not delay fusion.

## 1. Why this work — the cross-validation already changed the target

A five-scout literature review (all arXiv IDs verified) found:

- **Claim A for encoder-based VLMs: HIGH confidence, independently replicated.** Convergent
  mid-stack localization across Kaduri 2411.17491, Zhang 2411.18620, Neo 2410.07149,
  "Devils in Middle Layers" 2411.16724; DeepInsert 2504.19327 is causal confirmation (skip the
  early layers entirely, no loss). Causal-mediation (NOTICE 2406.16320) and VisiPruner 2510.17205 agree.
- **Claim A for NATIVE models: LOW–MEDIUM, with a direct counter-example.** The only
  mechanistic fusion-localization in native models — **The Narrow Gate (2412.06646)** — finds
  Chameleon/Emu3 keep modalities nearly orthogonal across all layers and route image→text through
  a single early **[EOI] "narrow gate"** (layers 2–6), *not* distributed mid-stack. No published
  work runs a clean causal fusion-depth probe on a native VLM the way we do. → **This is the
  contested claim the new experiments must adjudicate.**
- **Scale-migration sub-story (1.5→NeXT→OneVision) is our own contribution**, not literature
  consensus (consistent in spirit with Information Horizon 2512.07580). Treat as novel, hedge.
- **Claim B: corroborated** (NEO pre-Buffer 2510.14979, Mono-InternVL 2410.08202, EVE-2, BREEN),
  but the native *per-layer visual-expert* design (Mono-InternVL/BREEN = our `feat/visual-ffn-expert`
  branch) is the case where "before fusion" can blur into "distributed across all layers" → test it.
- **Methodology: attention-knockout is sound-with-caveats.** Self-repair/Hydra (2307.15771) means
  knockout *underestimates* importance (our windowed [0..d) design partly mitigates); add a
  **mean-ablation** variant (2509.17588) to kill the OOD-shock objection. "Attention mass ≠ fusion"
  is strongly supported (FastV 2403.06764, Visual Attention Sink 2503.03321, Massive Activations
  2402.17762). Recommended **orthogonal** triangulation metric: **residual-stream activation
  patching of image tokens** — perturbs the value/residual pathway, not attention edges, so
  self-repair and sink artifacts cannot reproduce its signal.

## 2. Goals & success criteria

- **G1 — External cross-validation.** Write `neo_report/CROSS_VALIDATION.md`: literature verdict
  per claim (with the contested-native finding and the novelty of the scale-migration story made
  explicit) + methodology audit. *Done in research; needs write-up.*
- **G2 — Orthogonal metric triangulation.** Add residual-stream **activation patching** (+ a
  **mean-ablation knockout** robustness variant) and run on existing key models
  {LLaVA-1.5, Qwen2.5-VL, NEO, SAIL} and all new models. Success = patching reproduces the φ
  fusion-depth ordering (or, if it disagrees, the disagreement is characterized).
- **G3 — Architecture generalization via controlled contrasts.** Run the full causal pipeline on
  the §3 slate, designed as within-family contrasts: **dense↔MoE** (InternVL3.5 8B vs 30B-A3B;
  Gemma-4 12B vs 26B-A4B) tests whether MoE relocates fusion; **within-recipe size scaling**
  (LLaVA-OV-1.5 4B vs 8B) + the recipe ladder (OV-7B→OV-1.5→OV-2) tests the scale-migration claim
  cleanly; Mono-InternVL stresses Claim B. Success = each headline claim is corroborated across the
  contrasts or qualified with evidence. (The direct Narrow-Gate 2412.06646 replication on
  Chameleon/Emu3 is deferred — re-added only if the native results are ambiguous.)
- **G4 — Benchmark/task generalization.** Add MMStar (MCQ, vision-indispensable → stronger R₀)
  and GQA (generative, gold-answer NLL). Success = report whether fusion depth is benchmark-
  invariant (MCQ→MCQ) and task-invariant (MCQ→generative) on a diagnostic model subset.

Non-goals: retraining any model; grounding/segmentation tasks; exhaustively re-running every
existing model on the new benchmarks (diagnostic subset only).

## 3. Model slate (final — within-family controlled contrasts; all HF ids verified)

Organized around **controlled comparisons** (hold the recipe constant, vary one axis) rather than
cross-family breadth — this is methodologically stronger for the generalization claims. All ids
verified via `hub_repo_details` + raw `config.json`; eager/4D-mask verified against the installed
transformers 5.10.2 source. None of the kept models use DeepStack / out-of-path injection.

**A. Does MoE / expert-routing relocate fusion? (Claim A under MoE — within-family dense↔MoE)**

| dense arm | MoE arm | family | why clean |
|---|---|---|---|
| `OpenGVLab/InternVL3_5-8B-HF` (Qwen3-8B, 36L, id 151671) | `OpenGVLab/InternVL3_5-30B-A3B-HF` (Qwen3-MoE 128/top-8, ~3B act, 48L) | InternVL3.5 (encoder) | same `internvl` HF class, same image-token id, plain causal mask — only dense↔MoE differs |
| `google/gemma-4-12B-it` *(existing baseline, dense)* | `google/gemma-4-26B-A4B-it` (sparse MoE 128/top-8, ~4B act, 30L, id 258880) | Gemma-4 (encoder-free) | same `gemma4` class + image-token id; only `enable_moe_block` differs; bidirectional-vision 4D mask |

**B. Does fusion depth scale with size / newer recipe? (the scale-migration claim — within-recipe)**

| models | tests |
|---|---|
| `lmms-lab/LLaVA-OneVision-1.5-4B-Instruct` vs `…-1.5-8B-Instruct` (Qwen3, self-contained TRC, id 151655) | within-recipe **size** scaling |
| `lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct` (newest, 2026-06-03; `llava_onevision2`, TRC) | newest recipe point; with existing OneVision-7B (2024) + OV-1.5 → a clean recipe ladder |

**C. Native per-layer visual expert (Claim B stress test)**

| `OpenGVLab/Mono-InternVL-2B` (InternLM2-1.8B, `internvl_chat` TRC, eager when no flash-attn) | vision built at *every* layer; = our `feat/visual-ffn-expert` arch — where "before fusion" may blur into "distributed" |

**D. Unified any-to-any (the class selected; BAGEL infeasible → Janus-Pro)**

| `deepseek-community/Janus-Pro-7B` (native `janus`, eager, in-seq id 100594, 30L) | feasible unified/image-gen rep; probe the understanding path (`generation_mode="text"`) |

**Cut / deferred (with reason — do not burn GPU):**
- **BAGEL `ByteDance-Seed/BAGEL-7B-MoT` — cut (infeasible as-is).** Hub repo has no `.py`/`auto_map`
  (`library: bagel-mot`, code in external `bytedance-seed/BAGEL` fork → violates no-fork rule);
  MoT path uses packed varlen attention, no 4D mask, no image-token id. Multi-day port; not worth it.
- **Qwen3-VL (`Qwen3-VL-30B-A3B-Instruct` / `-8B`) — cut.** DeepStack `[8,16,24]` injects visual
  embeds into the residual stream *outside* attention → mask surgery can't isolate the visual stream.
- **Chameleon-7B / Emu3-Chat-hf — deferred** (per user). Re-add for the direct Narrow-Gate
  (2412.06646) adjudication if the within-family / Mono-InternVL native results are ambiguous.
- **ERNIE-4.5-VL-28B-A3B / Kimi-VL-A3B / InternVL3.5-GPT-OSS-20B-A4B — optional.** Feasible but
  redundant given the cleaner within-family pairs; add only to widen.

**New core = 7 models:** InternVL3.5 {8B, 30B-A3B}, Gemma-4-26B-A4B, LLaVA-OV-1.5 {4B, 8B},
LLaVA-OV-2-8B, Mono-InternVL-2B (+ Janus-Pro-7B for the unified class = 8).

**Existing models retained as baselines** (already in `neo_report/`, supply the dense/recipe
anchors): NEO1.0/1.5 ×{2B,9B}×{MT,SFT}, Gemma-4-12B (dense anchor for the Gemma pair), SAIL-7B,
LLaVA-1.5-7B, LLaVA-NeXT-7B, LLaVA-OneVision-7B, Qwen2.5-VL-7B.

## 4. Metrics & probes

Existing (reused unchanged): **φ fusion-depth** (prefix + suffix + lastrow attention knockout,
`fusion_window.py`), **maturation strips** u_img/u_txt (`pathway_maturation.py`), **stream-freeze**
necessity (`freeze_probe.py`), **CKA**-to-DINOv2 (`cka_extract.py`+`cka_compute.py`).

New:
- **N1 — Residual-stream suffix mean-ablation (primary triangulation). ✅ BUILT & VALIDATED (P0,
  2026-06-16).** For cut depth `d`, replace each image token's residual with the per-image mean over
  image tokens in layers `[d..N)` (content intact for `[0..d)`, destroyed after). Retained
  `(acc_real − acc_donor)/R₀` rises 0→1; the marginal rise localizes the read onset. *Self-aligned*
  (each forward ablates its own tokens → works for native-res variable token counts) and orthogonal
  to attention-knockout (value pathway, not edges). **Validated on all 4 anchors — reproduces the
  knockout φ ordering** (LLaVA 0.22 vs φ0.28; Qwen 0.46/0.54; NEO 0.45/0.54; SAIL 0.56/0.59). Probes:
  `devtools/activation_patch.py` (+ neo/sail variants), analysis `devtools/patch_analysis.py`, figure
  `devtools/fig_patch_triangulation.py`. *Lesson:* the naive cumulative-prefix `[0..d)` version is
  non-discriminative (early content destruction propagates → every model saturates at layer 1);
  caught by gating on known-answer models. Cited: ROME 2202.05262, causal mediation 2004.12265,
  VLM precedent NOTICE 2406.16320.
- **N2 — Mean-ablation knockout robustness.** Re-run the φ knockout replacing the −∞ block with
  the **mean over image-token keys** (per 2509.17588). If the φ curve is stable to zero-vs-mean,
  the OOD-shock objection is answered. Small change to the mask-surgery path.
- **N3 (optional) — Tuned-lens read-out depth** (2303.08112) at the answer position, to cross-check
  the φ "completion" depth with a decode-side measurement.

Report each finding with the IOI circuit-evaluation framing where relevant (faithfulness /
completeness / minimality, 2211.00593).

## 5. Benchmarks

| Benchmark | Type | Scoring | Role | Coverage |
|---|---|---|---|---|
| VMCBench dev (existing) | MCQ (20×50) | letter logit | baseline | all models |
| **MMStar** (new) | MCQ, vision-indispensable | letter logit | dataset-generalization; larger R₀ | diagnostic subset + all new |
| **GQA** (new) | short-answer generative | gold-answer NLL, donor-swap R₀ | task-generalization (MCQ→free-form) | diagnostic subset |

Diagnostic subset for the new benchmarks (avoid re-running all ~17 models): one early-fuser
(LLaVA-1.5), one mid-fuser encoder (Qwen2.5-VL), one native (NEO or SAIL), plus the Narrow-Gate
pair (Chameleon, Emu3) and one new MoE/encoder. Enough to test invariance without full re-runs.

## 6. Engineering plan (units, each independently testable)

1. **`activation_patch.py`** (model-agnostic probe). New `kind`-parameterized probe sharing
   `fusion_window.py`'s loader/scoring/`find_layers`; adds residual `forward_hook` for image-token
   patching + the N2 mean-ablation switch. Smoke on an existing kind (llava) first.
2. **New `kind` branches** in `fusion_window.py`, `pathway_maturation.py`, `freeze_probe.py`
   (the last currently covers only llava/qwen/gemma — extend). Each branch: loader (eager), chat
   template, image-token id, verify dense 4D mask materializes.
   - *Native-`transformers` classes* (cleanest): `InternVL3_5-8B-HF` + `-30B-A3B-HF` (`internvl`),
     `gemma-4-26B-A4B-it` (extends the existing `gemma` kind), `Janus-Pro-7B` (`janus`).
   - *Self-contained trust_remote_code* (kind branch, load via `AutoModel`/the bundled class):
     `LLaVA-OneVision-1.5-4B/8B` (self-contained, no external pip import) and
     `LLaVA-OneVision-2-8B` (`llava_onevision2`, source-verify eager mask on first smoke).
3. **Custom-code analysis dir** for **Mono-InternVL-2B** (older `internvl_chat` `AutoModel`+TRC;
   mirror `neo_analysis/`/`sail_analysis/`): vendor modeling, force eager + dense 4D mask, per-model
   fusion/freeze/cka/pathway scripts. (The MoE arms are native-eager so a `kind` branch suffices.)
4. **Benchmark loaders**: MMStar (MCQ, reuse letter scoring) and GQA (generative, NLL of gold
   answer with donor-swap R₀) — small modules behind the existing prompt/scoring interface.
5. **Analysis/figures**: extend `window_analysis.py`, `fig_two_conclusions.py`, `fig_dol_pretty.py`
   to include new models; new activation-patching-vs-knockout overlay figure.

**Per-model validation invariants (block release of any model's numbers):** print
`N_layers / img_id / n_vis / letter_ids`; `assert n_vis>0`; `R₀>0.05`; `suf[0] ≡ cost[N-1]`
(bit-identical); `retained_suf(0)≈0`; intact accuracy within range of the model's published
benchmark number. (These caught real bugs before — stratified subset, letter-id space piece, etc.)

## 7. Compute & sequencing

SLURM on `ckpt-all` (l40s/l40/a100/h200), accounts `cse-ckpt`/`krishna-ckpt`. ~3–6 GPU-h per
7–8B model full pipeline; ERNIE-28B and Emu3-fp32 need a100/h200. Activation-patching adds ~1
sweep per model; benchmarks add diagnostic-subset runs. Estimated **~50–80 GPU-h, ~3–5 days**
wall-clock parallelized.

Phases (each gated by a smoke check):
- **P0** — N1 activation-patch probe + N2 mean-ablation, smoke + run on existing key models
  (LLaVA-1.5, Qwen2.5-VL, NEO, SAIL). *Triangulates the original headline first.*
- **P1** — Native-`transformers` + self-contained-TRC new models (InternVL3.5 8B & 30B-A3B,
  Gemma-4-26B-A4B, LLaVA-OV-1.5 4B & 8B, LLaVA-OV-2-8B, Janus-Pro-7B): kind branches + full
  pipeline + activation patch. Run each dense↔MoE pair back-to-back for the controlled contrast.
- **P2** — Mono-InternVL-2B custom dir + full pipeline.
- **P3** — MMStar + GQA on the diagnostic subset.
- **P4** — Analysis, figures, `CROSS_VALIDATION.md`, revised/qualified conclusions section.

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Eager 4D mask not materialized on a custom model → silent no-op flat curve | invariant checks (suf[0]≡cost[N-1], hook-hit counter, n_vis); per-model smoke before full run |
| 30B-A3B / 26B-A4B MoE memory | a100/h200 + `expandable_segments`; low active params keep compute fast; reduce n_causal if needed |
| Image-token id quirks (Mono `internvl_chat` runtime-set id) | resolve id at runtime from the model's own mechanism, print + assert n_vis |
| Dense↔MoE pair must share everything but the MoE block | verify identical image-token id + chat template across the pair before trusting the contrast (both InternVL3.5 = 151671, both Gemma-4 = 258880) |
| LLaVA-OV-2 TRC eager not source-verified | source-check the bundled `llava_onevision2` attention on first smoke; fall back to OV-1.5-8B if no 4D mask |
| Activation patch disagrees with knockout | characterize and report — disagreement between orthogonal causal methods is itself informative |
| Native generalization ambiguous after the contrasts | re-add the deferred Chameleon/Emu3 Narrow-Gate (2412.06646) replication as a follow-on phase |

## 9. Deliverables

- `neo_report/CROSS_VALIDATION.md` — literature + methodology verdict (with verified arXiv IDs,
  the contested-native finding, and the scale-migration novelty flagged).
- Extended φ table + division-of-labor figure including the 6 new models.
- Activation-patching-vs-knockout triangulation figure (existing key models + new).
- MMStar/GQA benchmark-invariance table on the diagnostic subset.
- Revised conclusions in `neo_report/`: the dense↔MoE contrast verdict (does MoE relocate
  fusion?), the within-recipe size-scaling verdict on the scale-migration claim (marked as our
  novel, internally-supported contribution), and the Claim B verdict from Mono-InternVL.
