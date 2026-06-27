# P0 — Activation-patch triangulation metric: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a residual-stream **activation-patching** probe (+ a mean-ablation variant) that localizes vision→text fusion depth *independently of attention*, and use it to triangulate the existing attention-knockout φ result on the four anchor models (LLaVA-1.5, Qwen2.5-VL, NEO, SAIL).

**Architecture:** Mirror `devtools/freeze_probe.py` exactly (same loader, letter scoring, donor `(i+37)%n` pairing, stratified causal subset, resume-safe jsonl, `find_layers`, `attn_implementation="eager"`). The probe registers one `forward_hook` per decoder layer that overwrites the **image-token positions' layer output**. Two intervention modes: `denoise` (run donor image, inject the clean image's cached per-layer image-token outputs in layers `[0..d)` → recovery curve from swap→intact; rise = sufficiency onset) and `meanabl` (run clean image, replace image-token outputs in `[0..d)` with their mean over image tokens → necessity curve; fall = where image-specific info is needed). Because it perturbs the value/residual pathway, not attention edges, self-repair (Hydra) and attention-sink artifacts cannot reproduce its signal — the core of the cross-validation argument.

**Tech Stack:** Python, PyTorch, HuggingFace transformers 5.10.2 (main `.venv`); NEO/SAIL in the `neo` venv. VMCBench dev via `datasets`. SLURM (`ckpt-all`, L40S/A100). Validation is empirical-invariant smoke tests, not pytest.

---

## File structure

- Create `devtools/activation_patch.py` — the HF probe (kinds: llava, qwen, gemma). Self-contained, ~150 lines, structured like `freeze_probe.py`.
- Create `devtools/activation_patch.slurm` — SLURM launcher.
- Create `devtools/patch_analysis.py` — reads `results_patch_*.jsonl`, computes the patch-localization quantiles (q10/q25/q50/q75/q90) and overlays them on the φ quantiles from `window_analysis.py` / `window_results.json`.
- Create `neo_analysis/neo_activation_patch.py`, `sail_analysis/sail_activation_patch.py` — NEO/SAIL variants (custom modeling, same intervention logic).
- Modify nothing in the existing probes (additive only).

---

### Task 1: Create the activation-patch probe (HF kinds)

**Files:**
- Create: `devtools/activation_patch.py`

- [ ] **Step 1: Write the probe.** Mirror `freeze_probe.py` for the shared parts (the helpers `load_vmcbench`, `doc_to_prompt`, `find_layers`, and the `__init__` loader block + `_build`/`_logits`/`_score` are copied verbatim from `freeze_probe.py` lines 34-130 — keep them identical so behavior matches). Replace the freeze hooks with the patch hook and `run()` below.

```python
"""ACTIVATION-PATCH probe: orthogonal causal triangulation of attention-knockout.
Intervenes on the IMAGE-TOKEN RESIDUAL STREAM (value pathway), not attention edges,
so self-repair / attention-sink artifacts cannot reproduce its signal.

  denoise@d : run DONOR image; overwrite image-position outputs in layers [0..d)
              with the CLEAN image's cached outputs. retained_denoise(d)/R0 rises
              from ~0 (d=0 == swap) to ~1 (d=N == intact); rise = sufficiency onset.
  meanabl@d : run CLEAN image; replace image-position outputs in [0..d) with their
              mean over image tokens (kills image-specific info, keeps norm).
              retained_meanabl(d)/R0 falls; fall = where image-specific info needed.

Usage: python devtools/activation_patch.py <model_id> <kind> <out.jsonl> [n] [n_causal] [mode]
  kind in {llava, qwen, gemma}; mode in {denoise, meanabl, both} (default both)
"""
import json
import sys
from pathlib import Path
import torch

DEV = "cuda"
POST = "Answer with the option's letter from the given choices directly.\n"

# load_vmcbench, doc_to_prompt, find_layers : COPY VERBATIM from freeze_probe.py (lines 34-57)

class PatchProbe:
    def __init__(self, model_id, kind):
        # COPY the loader block from freeze_probe.py __init__ (lines 62-80):
        #   AutoProcessor, kind->class, M.from_pretrained(..., attn_implementation="eager"),
        #   self.img_id, self.tok, self.letter_ids, self.layers, self.N
        ...
        self.mode = "none"   # none | capture | denoise | meanabl
        self.depth = 0       # patch layers [0..depth)
        self.pos = None      # bool (S,) image positions of the CURRENT run
        self.cache = {}      # layer i -> clean image-position outputs [n_vis, D]
        for i, lyr in enumerate(self.layers):
            lyr.register_forward_hook(self._post(i), with_kwargs=True)

    def _post(self, i):
        def f(_m, _args, _kwargs, out):
            if self.mode == "none":
                return None
            h = out[0] if isinstance(out, tuple) else out
            if self.mode == "capture":
                self.cache[i] = h[0, self.pos].detach().clone()
            elif self.mode == "denoise" and i < self.depth and i in self.cache:
                h[0, self.pos] = self.cache[i].to(h.dtype)
            elif self.mode == "meanabl" and i < self.depth:
                h[0, self.pos] = h[0, self.pos].mean(0, keepdim=True)
            return None
        return f

    # _build, _logits, _score : COPY VERBATIM from freeze_probe.py (lines 109-130)

    @torch.no_grad()
    def run(self, doc, donor, do_causal, depths, modes):
        gt = str(doc["answer"]).strip()
        if gt not in "ABCD":
            return dict(skip="bad_gt")
        q = doc_to_prompt(doc)
        bI, vmI = self._build(doc["image"].convert("RGB"), q)
        bIp, vmIp = self._build(donor["image"].convert("RGB"), q)
        rec = dict(gt=gt, category=doc.get("category"), n_vis=int(vmI.sum()),
                   seq_len=int(bI["input_ids"].shape[1]))
        self.mode = "none"
        rec["intact"] = self._score(self._logits(bI), gt)
        rec["swap"] = self._score(self._logits(bIp), gt)
        if not do_causal:
            return rec
        if int(vmI.sum()) != int(vmIp.sum()):
            rec["skip_causal"] = "nvis_mismatch"      # patch needs aligned image positions
            return rec
        rec["depths"] = depths
        if "denoise" in modes:
            self.mode, self.pos, self.cache = "capture", vmI, {}
            _ = self._logits(bI)                      # cache clean image-position outputs
            out = []
            for d in depths:
                self.mode, self.depth, self.pos = "denoise", d, vmIp
                out.append(self._score(self._logits(bIp), gt))
            rec["denoise"] = out
        if "meanabl" in modes:
            out, out_null = [], []
            for d in depths:
                self.mode, self.depth, self.pos = "meanabl", d, vmI
                out.append(self._score(self._logits(bI), gt))
                self.pos = vmIp
                out_null.append(self._score(self._logits(bIp), gt))
            rec["meanabl"], rec["meanabl_null"] = out, out_null
        self.mode, self.cache = "none", {}
        return rec


@torch.no_grad()
def main():
    model_id, kind, out_path = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
    n_causal = int(sys.argv[5]) if len(sys.argv) > 5 else 200
    mode = sys.argv[6] if len(sys.argv) > 6 else "both"
    modes = ("denoise", "meanabl") if mode == "both" else (mode,)
    ds = load_vmcbench()
    n = min(n, len(ds))
    stride = max(1, n // max(n_causal, 1))
    causal_set = set(range(0, n, stride))
    done = sum(1 for _ in open(out_path)) if out_path.exists() else 0
    pr = PatchProbe(model_id, kind)
    depths = list(range(0, pr.N + 1))                 # [0..N]: 0==swap/intact, N==intact/flattened
    print(f"[patch] {model_id} kind={kind} N={pr.N} modes={modes} n={n} "
          f"nc={len(causal_set)} resume={done}", flush=True)
    print(f"[patch] img_id={pr.img_id} letter_ids={pr.letter_ids}", flush=True)
    f = open(out_path, "a")
    for i in range(done, n):
        doc, donor = ds[i], ds[(i + 37) % n]
        try:
            rec = pr.run(doc, donor, do_causal=(i in causal_set), depths=depths, modes=modes)
            rec["i"] = i
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); rec = dict(i=i, skip="oom")
        except Exception as e:  # noqa: BLE001
            rec = dict(i=i, skip=f"{type(e).__name__}: {e}")
        f.write(json.dumps(rec) + "\n")
        if (i + 1) % 25 == 0:
            f.flush(); torch.cuda.empty_cache(); print(f"[patch] {i + 1}/{n}", flush=True)
    f.close()
    print("[patch] done", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax check (no GPU).**

Run: `python -c "import ast; ast.parse(open('devtools/activation_patch.py').read()); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit.**

```bash
git add devtools/activation_patch.py
git commit -m "feat(devtools): activation-patch probe (denoise + meanabl) for fusion-depth triangulation"
```

---

### Task 2: Invariant smoke on LLaVA-1.5 (GPU)

The "tests" are empirical invariants that MUST hold or the metric is broken. Requires a GPU
(see hyak-slurm-gpu skill; interactive L40S is enough for n=8).

**Files:**
- Test output: `/tmp/patch_smoke_llava.jsonl`

- [ ] **Step 1: Run the smoke.**

Run: `python devtools/activation_patch.py llava-hf/llava-1.5-7b-hf llava /tmp/patch_smoke_llava.jsonl 8 8 both`
Expected: prints `N=32`, `img_id=32000`, `letter_ids={...}`, finishes 8/8 with no `skip` on the causal items.

- [ ] **Step 2: Check the four invariants** with this inline script:

```python
import json
rows = [json.loads(l) for l in open('/tmp/patch_smoke_llava.jsonl')]
cz = [r for r in rows if 'denoise' in r]
assert cz, "no causal rows — check n_causal / nvis_mismatch skips"
for r in cz:
    dn, N = r['denoise'], len(r['depths']) - 1
    # INV1 (EXACT): depth-0 denoise == swap — no layer patched, donor run reproduced bit-for-bit
    assert dn[0]['nll'] == r['swap']['nll'], ("INV1 fail", r['i'])
    # INV3 (EXACT): meanabl depth-0 == intact — no ablation, clean run reproduced
    assert r['meanabl'][0]['nll'] == r['intact']['nll'], ("INV3 fail", r['i'])
# INV2 (APPROX): depth-N denoise recovers most of R0 toward intact (layer-0 text-attention
# leakage prevents exact equality, but retained must be high). Average over causal rows:
import statistics
R0 = statistics.mean(int(r['intact']['pred']==r['gt']) - int(r['swap']['pred']==r['gt']) for r in cz)
# use NLL margin as a continuous recovery measure (robust on n=8): denoise[N] NLL ~ intact NLL
rec = statistics.mean((r['swap']['nll'] - r['denoise'][N]['nll']) /
                      max(r['swap']['nll'] - r['intact']['nll'], 1e-6) for r in cz)
print("INV1,INV3 pass on", len(cz), "rows; INV2 mean NLL-recovery@N =", round(rec,2),
      "(want >0.8); n_vis:", [r['n_vis'] for r in cz])
assert rec > 0.8, "INV2 fail: full denoise patch does not recover the clean answer — broken hook"
```

Expected: `INV1,INV3 pass ... INV2 mean NLL-recovery@N = ~0.9...`. If `nvis_mismatch` skips dominate, LLaVA-1.5 is fixed-576 so it should NOT mismatch — investigate the `_build`/img_id before proceeding.

- [ ] **Step 3: Commit (smoke is throwaway; commit nothing, or note results in the task log).**

---

### Task 3: Full LLaVA-1.5 run + benchmark sanity (GPU/SLURM)

**Files:**
- Create: `devtools/activation_patch.slurm` (copy `devtools/freeze_probe`-style slurm header: `ckpt-all`, `--gres=gpu:l40s:1`, `--requeue`, the `neo`/main venv activation, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; body = the `python devtools/activation_patch.py "$MODEL" "$KIND" "$OUT" 1000 200 both` line parameterized by env vars).
- Output: `neo_analysis/results_patch_llava.jsonl`

- [ ] **Step 1: Launch.**

Run: `MODEL=llava-hf/llava-1.5-7b-hf KIND=llava OUT=neo_analysis/results_patch_llava.jsonl sbatch devtools/activation_patch.slurm`
Expected: a job id; on completion `1000/1000`, ~200 causal rows.

- [ ] **Step 2: Sanity-check intact accuracy** (catches the stratified-subset / letter-id bugs that bit the earlier study):

```python
import json
rows = [json.loads(l) for l in open('neo_analysis/results_patch_llava.jsonl') if 'intact' in json.loads(l)]
acc = sum(r['intact']['pred']==r['gt'] for r in rows)/len(rows)
print("intact acc:", round(acc,3))   # EXPECT ~0.52 (LLaVA-1.5 published VMCBench ~51.8)
```

Expected: `intact acc: ~0.52`. If far off, STOP and fix scoring before trusting any curve.

- [ ] **Step 3: Commit results.**

```bash
git add devtools/activation_patch.slurm neo_analysis/results_patch_llava.jsonl
git commit -m "data: activation-patch results for LLaVA-1.5 (P0 anchor)"
```

---

### Task 4: Patch-localization analysis + triangulation vs φ

**Files:**
- Create: `devtools/patch_analysis.py`
- Output: `neo_report/patch_results.json`, console table

- [ ] **Step 1: Write `patch_analysis.py`.** For each model jsonl: average over causal rows the
  `R0 = intact_acc - swap_acc`; `retained_denoise(d) = (acc_denoise(d) - swap_acc)/R0` (0→1 rising);
  `retained_meanabl(d) = (acc_meanabl(d) - acc_meanabl_null(d))/R0` (1→0 falling). Convert each to a
  normalized fusion-depth distribution exactly like `fig_two_conclusions.py::fusion_quants`:
  for denoise use `marg(d)=max(retained(d)-retained(d-1),0)` (the rise); for meanabl use
  `marg(d)=max(retained(d-1)-retained(d),0)` (the fall). Report q10/q25/q50/q75/q90 of each, in
  relative depth `d/N`. (Reuse the quantile helper from `fig_two_conclusions.py` — import or copy it.)

- [ ] **Step 2: Run and compare to φ.**

Run: `python devtools/patch_analysis.py neo_analysis/results_patch_llava.jsonl llava`
Expected: a row `llava | denoise q50=… | meanabl q50=… | knockout φ q50≈0.28`. The triangulation
claim succeeds if denoise/meanabl q50 land in the SAME region as the knockout φ (early for LLaVA-1.5);
record the numbers regardless — agreement validates, disagreement is itself a finding.

- [ ] **Step 3: Commit.**

```bash
git add devtools/patch_analysis.py neo_report/patch_results.json
git commit -m "feat(devtools): patch-vs-knockout triangulation analysis"
```

---

### Task 5: Qwen2.5-VL anchor (GPU/SLURM)

- [ ] **Step 1: Run.** `MODEL=Qwen/Qwen2.5-VL-7B-Instruct KIND=qwen OUT=neo_analysis/results_patch_qwen.jsonl sbatch devtools/activation_patch.slurm`
  - ⚠️ Qwen native-resolution → image-token count VARIES per image, so the donor `(i+37)` will
    often have a different `n_vis` → `nvis_mismatch` skips. Mitigation in `main()` for kind=qwen:
    set the processor to a FIXED `max_pixels`/`min_pixels` (e.g. `AutoProcessor.from_pretrained(..., max_pixels=512*28*28, min_pixels=512*28*28)`) so every image yields the same token count. Add this
    as a kind-conditional in `__init__`. Re-run; verify `nvis_mismatch` skip rate < 10%.
- [ ] **Step 2: Sanity** intact acc ≈ 0.76 (Qwen2.5-VL published).
- [ ] **Step 3: Analyze** `python devtools/patch_analysis.py neo_analysis/results_patch_qwen.jsonl qwen` → expect denoise/meanabl q50 ≈ mid-stack (knockout φ≈0.54).
- [ ] **Step 4: Commit results.**

---

### Task 6: NEO + SAIL anchors (neo venv, custom dirs)

**Files:**
- Create: `neo_analysis/neo_activation_patch.py` (copy `neo_analysis/neo_freeze_probe.py`, swap the freeze hook for the `_post` patch hook + `run()` from Task 1; keep NEO's loader/mask/img-id logic).
- Create: `sail_analysis/sail_activation_patch.py` (same, from `sail_analysis/sail_freeze_probe.py`).

- [ ] **Step 1:** Adapt NEO probe; smoke n=8 in the `neo` venv; check INV1-3.
- [ ] **Step 2:** Full NEO1.0-2B-SFT run + sanity (intact ≈ 0.71); analyze (expect mid-stack, φ≈0.54).
- [ ] **Step 3:** Adapt SAIL probe; smoke; full run (intact ≈ 0.73); analyze (expect mid-late, φ≈0.59).
- [ ] **Step 4:** Commit both probes + results.

---

### Task 7: Triangulation figure + P0 findings note

**Files:**
- Create: `neo_report/fig_patch_vs_knockout.png` (+ `.pdf`)
- Append to: `neo_report/CROSS_VALIDATION.md` (a "metric triangulation" subsection)

- [ ] **Step 1:** Plot, per anchor model, the knockout φ q50 vs denoise q50 vs meanabl q50 (with IQR
  whiskers) on one relative-depth axis; a diagonal = perfect agreement.
- [ ] **Step 2:** Write the verdict: do the two orthogonal residual interventions reproduce the
  attention-knockout fusion depth? (LLaVA-1.5 early, the other three mid/mid-late.) State the R0s and
  skip rates. This is the headline de-risking result before any new model is touched.
- [ ] **Step 3:** Commit.

---

## Subsequent phases (own plans, after P0 validates the metric)

P1 (new models — kind branches + full pipeline incl. activation_patch), P2 (Mono-InternVL custom
dir), P3 (MMStar + GQA), P4 (analysis/figures/CROSS_VALIDATION.md) follow
`docs/superpowers/specs/2026-06-15-vlm-fusion-generalization-design.md` §6–§9. Each gets its own plan
once P0 confirms the patch metric agrees with knockout (or characterizes the disagreement).

## Self-review notes
- Spec coverage: P0 implements spec §4-N1 (activation patching) + §4-N2 (mean-ablation) + the §2-G2
  triangulation on the four anchor models. P1-P4 deferred to their own plans (spec §6 phasing).
- The `nvis_mismatch` handling (Task 5) is the one real risk — fixed-token models (LLaVA, NEO fixed
  patch grid) are fine; variable-token models (Qwen) need the fixed-pixels mitigation. Flagged inline.
- Invariants INV1 (denoise[0]==swap) / INV2 (denoise[N]==intact) / INV3 (meanabl[0]==intact) are the
  correctness gates; a flat or inverted curve means a broken hook (catch before full runs).
