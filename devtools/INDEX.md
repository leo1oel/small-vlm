# devtools/ — fusion-study scripts

Only the fusion-study files are listed here. Everything else in this directory
(`sft_*`, `ckpt_*`, `lmms_*`, `infer_*`, `fa2_*`, `aux_fusion_*`, `budget_*`,
`text_capacity_*`, `xmodal_*`, `prefixlm_*`, `*train*`, lint/smoke, …) belongs
to the broader small-vlm training/eval project and is **left untouched**.

## Measurement probes (run on a model → jsonl)

| Script                    | Metric                                                     | Models (kind)                            |
| ------------------------- | ---------------------------------------------------------- | ---------------------------------------- |
| **pathway_maturation.py** | per-layer image/text stream work u_S(ℓ)                    | llava, qwen, gemma, onevision, llavanext |
| **fusion_window.py**      | fusion depth: prefix + suffix + lastrow attention knockout | same 5 kinds                             |
| **freeze_probe.py**       | stream-freeze necessity test                               | llava, qwen, gemma                       |
| **encoder_vlm_attn.py**   | FastV last→image attention                                 | llava, qwen, gemma                       |
| **cka_extract.py**        | per-layer image reps → npz (VLM or ref encoder)            | llava/qwen/gemma + dino/clip/siglip      |
| encoder_vlm_fusion.py     | original prefix-only probe (superseded by fusion_window)   | llava, qwen                              |

(NEO/SAIL equivalents live in `../neo_analysis/` and `../sail_analysis/`.)

## Analysis & figures (read jsonl/npz → tables + png/pdf)

| Script                          | Produces                                                       |
| ------------------------------- | -------------------------------------------------------------- |
| **window_analysis.py**          | window table, `fig_window.png`, `fig_lastrow.png`              |
| **freeze_analysis.py**          | `freeze_results.json`, `fig_freeze.png`                        |
| **fig_two_conclusions.py**      | ⭐ `fig_c1_fusion_depth`, `fig_c2_vision_building`             |
| **fig_dol_pretty.py**           | ⭐ `fig_division_of_labor_pretty` (Gill Sans)                  |
| **fig_prebuffer_cka.py**        | `fig_prebuffer_cka`                                            |
| **fig_freeze_bars.py**          | `fig_freeze_bars.png`                                          |
| cka_numbers.py / cka_compute.py | CKA tables / `fig_cka`                                         |
| analysis_master.py              | `master_results.json` (uses `fusion_crossmodel_figure.curves`) |
| fusion_crossmodel_figure.py     | **dependency** of analysis_master (keep)                       |

SLURM launchers: `pathway_mat.slurm`, `neo_pathway_mat.slurm`, `attn.slurm`,
`cka.slurm`, `neo_fusion_full.slurm`, `gemma4_fusion_full.slurm`.

`_archive/`: superseded figure scripts (fig_division_of_labor, encoderfree_figures,
fusion_density_figure, neo_stage_figure, neo_variants_figure).
