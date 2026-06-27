# VLM vision–text fusion study — overview & reading order (master index)

This is the entry point for the whole study. It now spans the original fusion-depth report plus three
follow-on programs: **P0** (cross-validation with an orthogonal causal metric), **P1** (generalization
across architectures and benchmarks, incl. WorldBench), and **P2** (verifying the "internal vision
encoder" claim). Each program also has a *Related work* section showing it does not contradict prior work.

## The headline conclusions

1. **Fusion is mid-stack — with or without a vision encoder, and it generalizes.** Vision is read into
   the answer in the LLM's middle layers (the only early case, LLaVA-1.5, is a small-scale frozen-encoder
   recipe, not its architecture). This holds across **architecture class** (dense, sparse-MoE in two
   families, unified, encoder-free), **size**, and **three benchmarks** — VMCBench, MMStar, and the harder,
   visually-diverse **WorldBench** (2000 MCQ, 7 domains) — and is even **invariant across visual domain**
   (q50 range 0.11–0.21 over objects/scenes/academics/agents). Confirmed by **two orthogonal causal
   metrics** (attention-knockout φ + residual-stream suffix mean-ablation), which kills the self-repair /
   attention-sink objections. Consistent with the interpretability-literature consensus (§ Related work).

2. **Visual features are built once, before fusion — and for native models the LLM grows its own encoder.**
   Wherever the capacity lives (external encoder, NEO's pre-Buffer, or interleaved early LLM layers), the
   visual representation is constructed before it is fused. For **encoder-free / native** VLMs this is now
   verified causally: their internal representation becomes **linearly mappable to a real encoder**
   (predictivity rises from a near-zero floor to encoder-grade), **classifies ImageNet** at the encoder
   ceiling (SAIL peak 0.82 > CLIP 0.77; ≫ its layer-0 floor 0.009), and a **random-init control** proves
   this is *learned by generative pre-training*, not architectural. Supported by prior work on encoder-free
   MLLMs and representational convergence (§ Related work).

## Documents (read top to bottom)

### Core findings
| File | What it is |
|---|---|
| **PAPER_ANALYSIS.md** | ⭐ Paper-ready section + figure captions for Conclusions 1 & 2. Start here for the original result. |
| **METHODS_HOWTO.md** | ⭐ Every metric defined/measured + exact commands to reproduce on a new model. Verified against code. |
| **FUSION_WINDOW_REANALYSIS.md** | Definitive original results: fusion window, pathway decomposition, freeze experiment, bridge models. |

### Cross-validation & generalization (P0–P2, latest)
| File | What it is |
|---|---|
| **CROSS_VALIDATION.md** | **P0** — external literature triangulation + the orthogonal residual-stream causal metric (`sufmeanabl`) reproducing the knockout fusion depth. |
| **GENERALIZATION_RESULTS.md** | **P1** — fusion depth across MoE/size/unified/encoder-free (11 models); **§2.6 WorldBench** (frontier + by-domain); **§5 Related work** (mid-stack = literature consensus). |
| **INTERNAL_ENCODER.md** | **P2** — Claim B verified: native VLMs grow an internal vision encoder (predictivity + ImageNet probe + random-init causal control); **§5.5 Related work**. |

### History / superseded
| File | What it is |
|---|---|
| ENCODER_FREE_FUSION_REPORT.md | v1 report (correction banner; superseded interpretation, correct numbers). |
| FUSION_METRICS_EXPLAINED.md | Earlier metrics explainer (zh); subsumed by METHODS_HOWTO. |

## The final figures

**⭐ The persuasive headline set (one clear conclusion each; `devtools/fig_persuasive.py` + `fig_internal_encoder.py`):**
| File | Shows |
|---|---|
| **fig_fusion_forest.png** | **Conclusion 1** — fusion is mid-stack across **10 architectures** (forest plot: IQR whiskers, family-coloured, MoE diamonds; LLaVA-1.5 omitted as the one explained outlier). |
| **fig_invariance.png** | **Conclusion 1 robustness** — fusion depth invariant **across 3 benchmarks** (VMCBench/MMStar/WorldBench, flat slopegraph) and **across 7 visual domains** (WorldBench, tight band). |
| **fig_triangulation.png** | **Credibility** — two *orthogonal* causal metrics (attention-knockout φ vs residual-stream sufmeanabl) rank all models identically (Pearson r=0.96), killing self-repair / attention-sink objections. |
| **fig_internal_encoder.png** | **Conclusion 2** — native predictivity rises to encoder-grade vs encoder-VLM flat vs random-init flat-low; ImageNet floor→peak bars (native peak ≥ encoder ceiling). |
| **fig_build_before_fuse.png** | **Connects 1 & 2** — on one depth axis, the visual representation reaches encoder-grade (predictivity build-up, unimodal) *before* the fusion interval reads it (clear for Gemma/NEO; SAIL is the tight borderline). The original `fig_division_of_labor_pretty.png` shows the same via image-stream work. |

**Supporting / original figures:**
| File | Shows |
|---|---|
| fig_c1_fusion_depth / fig_c2_vision_building / fig_division_of_labor_pretty.{png,pdf} | original Conclusion-1/2 box-whiskers + combined map. |
| fig_worldbench.png / fig_p1_generalization.png | earlier WorldBench + architecture-generalization panels (now subsumed by the headline set). |
| fig_prebuffer_cka.{png,pdf} | pre-Buffer builds vision-encoder-grade features (CKA vs DINOv2). |
| fig_window.png / fig_lastrow.png / fig_freeze_bars.png / fig_attention.png | onset window / answer-row pathway / freeze test / FastV attention. |

`_archive/` holds superseded v1 figures and the old NEO-pre-Buffer report. Nothing there is read by the live pipeline.

## Where the code + data live
- Probes & analysis: `../devtools/` (HF models), `../neo_analysis/` (NEO), `../sail_analysis/` (SAIL),
  `../mono_analysis/` (Mono-InternVL). See `../devtools/INDEX.md`.
- Raw results: `../neo_analysis/results_*.jsonl` (fusion/patch + `results_wb_*` WorldBench),
  `predictivity_*.json`, `imagenet_probe_results.json`, `wb_domain_*.json`, `cka_*.npz`.
