# Cross-validation of the fusion-depth study — external literature + an orthogonal causal metric

*2026-06-16. Answers the question "are the `neo_report/` conclusions reliable?" two ways: (1) a
five-scout review of the independent interpretability literature (all arXiv IDs verified), and
(2) a new, mechanistically-orthogonal causal metric (residual-stream **suffix mean-ablation**) run
on the four anchor models, which reproduces the attention-knockout fusion depth. Companion to the
generalization program in `docs/superpowers/specs/2026-06-15-vlm-fusion-generalization-design.md`.*

---

## 0. TL;DR

- **The encoder-side headline is solid and independently replicated** (HIGH confidence): four
  independent attention-knockout / attention-lens studies plus one architectural intervention
  (DeepInsert) converge on mid-stack answer-integration in encoder-based VLMs.
- **The native-model generalization — the report's actual novelty — is *contested* in the
  literature** (LOW–MEDIUM): the one mechanistic study of native models (The Narrow Gate, 2412.06646)
  finds Chameleon/Emu3 fuse via an *early* localized gate, not distributed mid-stack. **This is the
  claim the new-model experiments must adjudicate.**
- **The scale-migration sub-story (LLaVA-1.5 → NeXT → OneVision fuses progressively later) is the
  report's own contribution**, not literature consensus — present it as novel, hedge it.
- **The attention-knockout method is sound-with-caveats**; its "attention mass ≠ fusion" sub-claim
  is *strongly* supported by independent work.
- **New empirical result:** a value-pathway metric orthogonal to attention-knockout independently
  reproduces the fusion-depth ordering (LLaVA early, Qwen/NEO mid), which **kills the two most
  serious objections to the knockout result** (self-repair and attention-sink artifacts). And an
  independent recomputation of φ reproduced the report's published values (LLaVA 0.28, Qwen 0.54).

---

## 1. External literature — does the field agree?

### 1.1 Claim A (mid-stack fusion) for ENCODER-based VLMs — HIGH, replicated

| arXiv | method | finding |
|---|---|---|
| **2411.17491** Kaduri | attention knockout | cross-modal integration in the **middle ~25% of layers** (LLaVA-1.5 ~L4-20, InternVL2-76B ~L20-40/80) |
| **2411.18620** Zhang | attention knockout | two-stage; **object-specific** image→question transfer in **middle** layers (LLaVA-1.5/13B, v1.6, NeXT) |
| **2410.07149** Neo et al. | knockout + logit lens | object→last-token read strongest **mid-to-late (L15-24)**; logit-lens interpretable ~L25/33 |
| **2411.16724** Devils-in-Middle | attention lens | visual interaction **L5-26** (enrich L5-18, refine L19-26) |
| **2504.19327** DeepInsert | architectural | **insert visual tokens at a middle layer, skip early layers entirely → no loss** (causal proof early layers don't carry fusion) |
| **2406.16320** NOTICE | causal mediation | **middle-layer cross-attention heads** are causal |
| **2510.17205** VisiPruner | — | fusion "occurs abruptly in **middle layers**"; shallow layers = passive sinks |

**Verdict:** independent, methodologically diverse work converges on mid-stack integration for the
LLaVA/Qwen/InternVL family. The study's method measures what the field measures. *Honest flags:*
Basu 2406.04236 ("storage" is early) and "From Redundancy" 2406.06579 ("redundant early") are about
a *different* quantity (where info enters/becomes prunable), not answer-integration — do not cite as
mid-stack support. BLIP (Palit 2308.14179, L9-11) is a real later-fusion case but is an
encoder-decoder Q-Former, not a decoder-only stack.

### 1.2 Claim A for NATIVE / encoder-free VLMs — LOW–MEDIUM, a direct counter-example

The only mechanistic fusion-localization in native models is **The Narrow Gate (2412.06646)**: in
Chameleon and Emu3 the two modalities stay **nearly orthogonal across all layers**, and image→text
information funnels through a single **[EOI] "narrow gate"** used **early** (layers 2-6 in Chameleon).
That is a *different mechanism*, not a relocated mid-stack depth. **No published work runs a clean
causal fusion-depth probe on a native VLM the way this study does.** → The report's claim that
mid-stack fusion is architecture-independent and extends to native VLMs is a *hypothesis under test*,
with a named result pointing the other way. The new-model slate keeps Chameleon+Emu3 (deferred) for
exactly this adjudication.

### 1.3 Claim B (vision built before fusion) — corroborated

NEO's pre-Buffer (the native-VLM NEO paper is **2510.14979**), Mono-InternVL's per-layer visual
experts (**2410.08202**: "locality in shallow-layer visual encoding", "barely interactive at shallow
layers, gradually fused as layers deepen"), EVE-2 (2502.06788), BREEN (2503.12446) all build vision
in an early/parallel modality-specialized pathway. *Caveat:* the per-layer visual-expert design
(Mono-InternVL/BREEN = this repo's `feat/visual-ffn-expert` branch) is where "before fusion" may
blur into "distributed across all layers" — the Mono-InternVL probe in P1 tests this.

### 1.4 The scale-migration sub-story — the report's own, not consensus

No independent paper tests fusion-*depth* migration across LLaVA-1.5 → NeXT → OneVision. It is
consistent in spirit with "Information Horizon" (2512.07580: stronger models keep visual tokens
informative deeper) but should be presented as a novel contribution and hedged.

---

## 2. Methodology audit

**Attention-knockout is a legitimate causal-localization method** (Geva 2304.14767; circuit-eval
discipline from IOI 2211.00593), with disclosed caveats:
- **Self-repair / Hydra effect (2307.15771):** ablation under-estimates a layer's importance →
  observed degradation is a *lower bound*. The windowed `[0..d)` design partly mitigates (it ablates
  backups too). A **mean-ablation knockout** variant (per 2509.17588) would answer the OOD-shock
  objection.
- The **"attention mass ≠ fusion"** sub-claim is *strongly* supported: FastV (2403.06764), Visual
  Attention Sink (2503.03321, ICLR'25: high-attention image tokens are semantically irrelevant),
  Massive Activations (2402.17762), VisiPruner (2510.17205).
- **Recommended orthogonal triangulation:** intervene on the image-token *residual/value pathway*,
  not attention edges — self-repair and attention-sinks cannot reproduce its signal. This is the
  metric built and validated in §3.

---

## 3. New result — an orthogonal causal metric reproduces the knockout fusion depth

**Metric (suffix mean-ablation, `sufmeanabl`).** For cut depth `d`, replace each image token's
residual with the per-image mean over image tokens in layers `[d..N)` (image content intact for
`[0..d)`, destroyed after). The usable-image-signal retention `r(d) = (acc(real) − acc(donor))/R₀`
rises from 0 (`d=0`, all flattened) to 1 (`d=N`, none flattened); the marginal **rise localizes the
depth up to which image content must survive = the read onset**. It is *self-aligned* (each forward
ablates its own image tokens → no cross-image position matching, so it works for native-resolution
models with variable token counts) and *mechanistically orthogonal* to attention-knockout (value
pathway, not attention edges).

> *Methodological note worth keeping:* the naive **prefix** version (`[0..d)`, cumulative from layer
> 0) is **non-discriminative** — destroying/injecting image content early propagates forward, so
> every model saturates at layer ~1 (LLaVA and Qwen both gave q50≈0.03-0.04 despite φ = 0.28 vs 0.54).
> The suffix formulation fixes this. This was caught by gating on known-answer models before scaling.

**Result (VMCBench dev, n_causal≈150-200, letter scoring, donor (i+37) pairing):**

| Model | family | `sufmeanabl` q50 [q25,q75] (NEW, value-pathway) | knockout φ q50 (attention) | agree? |
|---|---|---|---|---|
| LLaVA-1.5-7B | enc | **0.22** [0.22, 0.34] | 0.28 | both **early** |
| Qwen2.5-VL-7B | enc | **0.46** [0.29, 0.54] | 0.54 | both **mid** |
| NEO1.0-2B-SFT | native | **0.45** [0.30, 0.60] | 0.54 | both **mid** |
| SAIL-7B | native | **0.56** [0.47, 0.66] | 0.59 (suffix-onset) | both **mid-late** (latest of the four) |

The orthogonal metric reproduces the knockout ordering (LLaVA early ≪ Qwen/NEO mid); `sufmeanabl`
onset sits ~0.06-0.09 *earlier* than φ, as expected (onset of the read vs. its completion).
*(Figure: `fig_patch_vs_knockout.{png,pdf}`.)*

**Why this matters:** the two interventions fail in different ways, so agreement is strong evidence.
A value-pathway intervention cannot be rescued by attention self-repair (there is no attention edge
to back up) and is immune to attention-sink artifacts (it never reads attention mass). That the
mid-stack fusion depth survives both → the central claim is not an artifact of the knockout method.

---

## 4. What remains (the generalization program)

The literature pinpoints **where** the open question is: not "does mid-stack fusion hold for
encoder VLMs" (it does), but "does it extend to native / MoE / unified architectures, or do those
fuse differently (Narrow-Gate-style early gating)?" The P1 slate — dense↔MoE contrasts
(InternVL3.5, Gemma-4), the per-layer-visual-expert native (Mono-InternVL), recipe scaling
(LLaVA-OV-1.5 4B/8B, OV-2), and (deferred) the Narrow-Gate VQ models — is aimed exactly there, and
each new model will be triangulated with the `sufmeanabl` metric validated here. (P1 runs after the
June 16–23 cluster maintenance.)
