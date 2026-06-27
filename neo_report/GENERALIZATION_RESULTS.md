# P1 — Architecture-generalization results (rolling)

*Started 2026-06-17 (post-maintenance). Tests whether the mid-stack-fusion conclusion generalizes
across MoE / recipe-scaled / unified architectures, via within-family controlled contrasts, using
the P0-validated metrics: residual-stream **suffix mean-ablation** (`sufmeanabl`, read-onset) and
attention-knockout **φ**. Lower q50 = earlier fusion. See `CROSS_VALIDATION.md` §3 for metric
validation. **This file updates as runs land.***

---

## 0. TL;DR

- **MoE does NOT move fusion depth — confirmed in TWO families.** InternVL3.5 dense↔MoE (0.42↔0.46)
  and Gemma-4 dense↔MoE (0.42↔0.37): in both, the sparse-MoE arm fuses at the same mid-stack depth as
  its dense sibling. Mechanism: MoE swaps the per-token FFN, not the **shared attention** that carries
  the image→answer read.
- **Size barely moves it.** InternVL3.5 2B/4B/8B = 0.39/0.42/0.42 — stable mid-stack across a 4×
  parameter range (only slight deepening). So the report's recipe-scaling migration is a *recipe/data*
  effect, not a *parameter-count* effect.
- **All new architectures fuse mid-stack (0.37–0.46)** — encoder-MoE (×2 families), size-scaled — with
  the unified Janus-Pro (0.30) on the early side, matching the report's "decoupled contrastive encoder →
  LLM-ready features → earlier read" gradient (like LLaVA-1.5). The mid-stack conclusion generalizes.
- **The orthogonal metric holds on new models:** InternVL3.5-8B sufmeanabl 0.42 ≈ its knockout φ 0.44.
- **Fusion depth is benchmark-invariant.** On MMStar (harder, vision-indispensable MCQ) the q50s track
  VMCBench (LLaVA 0.22→0.19, Qwen 0.46→0.50, InternVL3.5-8B 0.42→0.47, InternVL-MoE 0.46→0.44) — same
  ordering, ±0.05. Fusion depth is a model property, not a dataset artifact.
- **…and holds on a frontier, visually-diverse benchmark + across visual domains (WorldBench).** On
  WorldBench (2000 MCQ, 7 dense/diverse visual domains) Gemma-4-12B q50=0.42, Qwen2.5-VL q50=0.50, and
  InternVL3.5-8B q50=0.42 again track their VMCBench/MMStar values (Qwen's 0.50 = its MMStar 0.50), and
  stratifying by visual domain the q50 stays in a tight band (range 0.11–0.21). Fusion depth is invariant
  to *what kind of image* too — not just which dataset. Honest boundary: dense-OCR domains (Documents/Charts,
  Digital World) give little/no causal signal (R₀≤0.04) at standard resolution; and the metric needs a model
  strong enough to use the image (LLaVA-1.5 is at chance → dropped). (§2.6)
- **Consistent with the literature, not contradicting it.** Independent interpretability work converges on
  mid-layer cross-modal integration (Cross-modal Information Flow 2411.18620; Narrow Gate 2412.06646;
  What's in the Image 2411.17491; etc.). Our novelty is two *orthogonal causal* metrics that agree, plus
  invariance across architecture/size/benchmark/domain. (§5)

## 1. Status

| Contrast | model | id | sufmeanabl q50 | φ q50 | intact | status |
|---|---|---|---|---|---|---|
| **MoE (encoder)** | InternVL3.5-8B (dense) | `OpenGVLab/InternVL3_5-8B-HF` | **0.42** [0.31,0.53] | **0.44** [0.39,0.56] | 0.81 | ✅ sufmeanabl≈φ |
| | InternVL3.5-30B-A3B (MoE) | `…-30B-A3B-HF` | **0.46** [0.29,0.62] | — | 0.82 | ✅ sufmeanabl |
| **MoE (enc-free)** | Gemma-4-12B (dense) | `google/gemma-4-12B-it` | **0.42** [0.25,0.50] | 0.48 (report) | 0.785 | ✅ |
| | Gemma-4-26B-A4B (MoE) | `google/gemma-4-26B-A4B-it` | **0.37** [0.20,0.43] | — | 0.775 | ✅ ≈dense |
| **Unified** | Janus-Pro-7B | `deepseek-community/Janus-Pro-7B` | **0.30** [0.23,0.43] | — | 0.645 | ✅ early-mid |
| **Size scaling** | InternVL3.5-2B-HF | `OpenGVLab/InternVL3_5-2B-HF` | **0.39** [0.21,0.46] | — | 0.735 | ✅ |
| | InternVL3.5-4B-HF | `…-4B-HF` | **0.42** [0.28,0.50] | — | 0.788 | ✅ |
| | (InternVL3.5-8B above) | | 0.42 | | 0.81 | ✅ |
| | ~~LLaVA-OV-1.5/2~~ | (TRC) | — | — | — | ❌ TRC 与 transformers 5.10 多处不兼容 (pad_token_id, KeyError 'default'); 用 InternVL size 阶梯替代 |
| **Native expert (Claim B)** | Mono-InternVL-2B | `OpenGVLab/Mono-InternVL-2B` | — | — | — | ⏸️ 降优先级: `InternVLChatModel`(internvl_chat), img token 运行时设, forward 非标准(pixel_values+IMG_CONTEXT), transformers 5.10 加载 `KeyError:'type'`(疑需 neo venv 4.57)。需 NEO 级自定义 dir。Claim B 文献已支持; 备选: 用本仓 feat/visual-ffn-expert 自训 ckpt 直接测 |

Baselines (P0, already mid-stack): LLaVA-1.5 0.22 (early), Qwen2.5-VL 0.46, NEO-2B 0.45, SAIL 0.56.

## 2. Findings (as they land)

### 2.1 Does MoE relocate fusion? — NO (confirmed in two families)
InternVL3.5 dense 8B (0.42) ≈ MoE 30B-A3B (0.46); Gemma-4 dense 12B (0.42) ≈ MoE 26B-A4B (0.37). In
both an encoder-MoE and an encoder-free-MoE family, the sparse arm fuses mid-stack like its dense
sibling (IQRs overlap heavily). Mechanistic reading: the MoE only swaps the per-token FFN sub-network;
the **shared attention** — the channel that carries the image→answer read — is unchanged, so fusion
depth is unchanged.

**Metric cross-check on a P1 model:** InternVL3.5-8B's `sufmeanabl` q50 (0.42) reproduces its *own*
attention-knockout φ (0.44) — the orthogonal residual metric tracks φ on a brand-new model too,
extending the P0 triangulation beyond the four anchors.

### 2.2 Size scaling — fusion is size-stable
InternVL3.5 2B/4B/8B = 0.39/0.42/0.42 (same recipe, 4× params): mid-stack throughout, only a slight
deepening with size. So the report's 1.5→NeXT→OneVision migration is a *recipe/data* effect, not a
*parameter-count* effect — a useful refinement of the (own, novel) scale-migration claim.
(OV-1.5/2 dropped: their trust_remote_code is incompatible with transformers 5.10 — `pad_token_id`,
`KeyError 'type'`; InternVL3.5 native sizes cover the size axis cleanly instead.)
### 2.3 Unified (Janus-Pro-7B) — early-mid (q50 0.30).
Janus probes its understanding path (decoupled SigLIP encoder → 576 fixed tokens). It fuses earlier
than the mid-stack cluster (0.30 vs 0.42–0.56), resembling LLaVA-1.5: a contrastively-aligned
encoder feeds LLM-ready visual features, so the answer reads them earlier (though not as early as
LLaVA-1.5's 0.22). Consistent with the report's "encoder LLM-readiness → earlier read" gradient.
### 2.4 Native per-layer visual expert (Claim B) — deferred (Mono-InternVL needs a custom dir; see table + §3).

### 2.5 Benchmark invariance — fusion depth tracks across datasets
On MMStar (1500 MCQ, curated vision-indispensable) the `sufmeanabl` q50 reproduces VMCBench within
±0.05 and preserves the ordering: LLaVA-1.5 0.22→0.19 (early), Qwen 0.46→0.50, InternVL3.5-8B
0.42→0.47, InternVL3.5-30B-A3B-MoE 0.46→0.44 (mid). The dense↔MoE equivalence also holds on MMStar
(0.47 ≈ 0.44). R₀ is healthier on MMStar for the stronger models (0.30–0.43; vision matters more) and
tiny for LLaVA-1.5 (0.065 — MMStar is hard for it). → **fusion depth is a model property, not a
benchmark artifact.** Results: `results_mmstar_{llava,qwen,ivl8,ivl30moe}.jsonl`.

### 2.6 WorldBench — fusion depth holds on a frontier, visually-diverse benchmark (+ domain-invariance)
WorldBench (`zlab-princeton/WorldBench`, arXiv 2606.06538; 2000 MCQ across **7 visually-distinct
domains** — Living Things, Objects, Scenes, Digital World, Academics, Documents/Charts/Tables, Agents)
is far harder and more visually diverse than VMCBench/MMStar (dense OCR, UI screenshots, charts). It is
the strongest cross-benchmark test, and it adds a *within-benchmark* axis the others lack: visual domain.

| model | WorldBench `sufmeanabl` q50 | (its VMCBench / MMStar) | R₀ | domain q50 range |
|---|---|---|---|---|
| Gemma-4-12B (full 2000) | **0.42** [0.27,0.58] | 0.42 / — | 0.106 | 0.12 |
| Qwen2.5-VL-7B @896 (1850) | **0.50** [0.29,0.64] | 0.46 / 0.50 | 0.145 | 0.21 |
| InternVL3.5-8B† (875) | **0.42** [0.36,0.53] | 0.42 / 0.47 | 0.082 | 0.11 |

† InternVL hit the 6 h wall clock at 875/2000 rows (native-res, large WorldBench images are slow), so its
R₀/curve rest on 219 causal rows — supporting, not primary. Gemma (full) and Qwen (1850) are the primary anchors.

**Fusion depth is still mid-stack on WorldBench** (q50 0.42–0.50), tracking each model's VMCBench/MMStar
value within ±0.05 — a third, much harder benchmark with the same answer (Qwen's 0.50 reproduces its MMStar
0.50 exactly; Gemma's 0.42 = its VMCBench 0.42). **And it is visual-domain-invariant:** stratifying by
domain, every signal-bearing domain's q50 sits in a tight band — InternVL range 0.11 (Objects 0.42, Living
0.42, Scenes 0.42, Academics 0.39, Agents 0.31), Gemma range 0.12, Qwen range 0.21 (slightly noisier). So
fusion depth does not move with the *kind* of visual content, only stays mid-stack. Notably, **at 896 px
Qwen recovers signal on Digital World** (R₀ +0.07, q50 0.36) that vanished at 448 px — confirming the earlier
dead-signal there was a resolution artifact, not a fusion-depth shift. Figure `neo_report/fig_worldbench.png`;
per-domain `neo_analysis/wb_domain_wb_{ivl8,gemma12,qwen}.json`.

**Two honest findings on the boundary of the method:**
- **The densest-OCR domain yields no causal signal.** Documents/Charts/Tables gives R₀ ≤ 0.04 for
  *every* model — the answer barely changes when the image is swapped/ablated, so no read-onset is
  definable. (Digital World was also dead at 448 px but **recovers** at 896 px for Qwen, R₀ +0.07 — so
  that one was a resolution artifact, not a true no-signal domain.) For the genuinely resistant
  Documents/Charts case, either current models do not reliably read it or the information is too diffuse
  to localize. This is a limit of what the metric can measure, not a counterexample to mid-stack fusion
  (where signal exists, it is mid-stack).
- **The metric requires a model competent at the benchmark.** The causal read-onset is only defined when
  the image matters (R₀ > 0.05). WorldBench is a frontier benchmark; **LLaVA-1.5 (2023) scores at chance
  and gets R₀ < 0** (the correct image helps no more than a random one), so no fusion depth is definable
  for it — it was dropped. Strong native-res models (InternVL3.5-8B, Gemma-4-12B; Qwen2.5-VL at 896px,
  run pending) are the valid anchors. Results: `results_wb_{ivl8,gemma12,qwen}.jsonl`.

## 3. Engineering notes
- New `kind` branches added & in use: `internvl`, `janus` (native classes); `onevision15`/`onevision2`
  (self-contained TRC via `AutoModelForImageTextToText`); `gemma4moe` (`Gemma4ForConditionalGeneration`).
- Native-res models (InternVL, OV, Janus-gen) → variable n_vis → denoise skips, `sufmeanabl` runs
  (self-aligned). Janus understanding path is fixed 576 (SigLIP).
- Downloads work from compute nodes (login node offline); `HFOFF=0` in the slurm enables them.
- Benchmark is parameterized: `BENCH` env var → `load_dataset_bench`; `doc_to_prompt` auto-adapts
  (VMCBench separate A/B/C/D fields vs MMStar embedded options).

## 4. Optional extensions (turnkey, await sign-off)
- **GQA (generative / task-type invariance — addresses the report's biggest caveat "MCQ-only").**
  Fields verified: `lmms-lab/GQA` config `testdev_balanced_instructions` has `question` / `answer`
  (short, e.g. "no") / `imageId` but **no image** — images live in config `testdev_balanced_images`
  (join by `imageId`). Needs (a) a `load_gqa` joining the two configs; (b) **generative scoring** to
  replace letter-logit scoring — gold-answer first-token NLL + first-token-argmax match as the
  VQA-accuracy proxy, R₀ = acc(real)−acc(donor). Design choice to confirm: first-token vs full-answer
  teacher-forced NLL. Larger probe change than MMStar (data join + scoring) — worth a quick sign-off.
- **Mono-InternVL-2B (Claim B direct test).** Needs a NEO-style custom dir (InternVLChatModel:
  runtime image-token id, non-standard forward, transformers-5.10 `KeyError 'type'` → neo venv).
  Claim B already supported by the literature; alternative: probe this repo's own
  `feat/visual-ffn-expert` checkpoint directly.

## 5. Related work — does "mid-stack fusion" contradict prior interpretability work? (No — it is the consensus)

We searched the VLM-interpretability literature for where visual information enters the language stream.
**The independent consensus is mid-layer cross-modal integration, which our result reproduces and extends.**

- **Cross-modal Information Flow in MLLMs** (arXiv 2411.18620): visual information is integrated into the
  language stream in **lower-to-middle layers**, then propagates to the answer position in mid-to-later
  layers. Directly consistent with our mid-stack read-onset.
- **From Redundancy to Relevance** (2406.06579): image tokens aggregate into a few "anchor" tokens in
  shallow-to-middle layers; text reads from them later. Same mid-stack locus, attention-flow method.
- **The Narrow Gate** (2412.06646): image→text communication funnels through a small set of **middle-layer**
  positions. Consistent (a sparse mid-stack channel).
- **What's in the Image?** (2411.17491) and **Towards Interpreting Visual Information Processing in VLMs**
  (2410.07149): image information is accessed primarily in **middle layers** via attention; later layers
  refine. Consistent.
- **What Do Visual Tokens Really Encode?** (2603.00510): internal visual processing is largely redundant
  except where **mid-layer injection** is effective on vision-intensive tasks. Consistent, and echoes our
  WorldBench finding that signal concentrates on vision-dependent items.

**No contradiction surfaced.** These works mostly use a *single* method (attention/information-flow or
logit-lens) on a couple of LLaVA-family models. Our contribution over them is (i) **two orthogonal causal
metrics** — attention-knockout φ *and* residual-stream suffix mean-ablation — that agree, ruling out
self-repair / attention-sink artifacts; and (ii) **invariance demonstrated across architecture class**
(dense, sparse-MoE in two families, encoder-free, unified), **size**, and now **three benchmarks**
(VMCBench, MMStar, WorldBench) **and seven visual domains**. The "language-prior / blindness" line (VLMs
sometimes answer without using the image) is also consistent: precisely on the WorldBench dense-OCR
domains where models cannot use the image, the causal signal R₀ vanishes.
