# Paper-ready analysis section (EN) + figure caption

## X. Where does vision–text fusion happen, and what do the layers before it do?

**Setup.** We analyze ten VLMs spanning two architecture families: six *native*
models that consume patch embeddings directly with no vision encoder (NEO-1.0/1.5
at 2B/9B, Gemma-4-12B, SAIL-7B) and four *encoder-based* models (LLaVA-1.5-7B,
LLaVA-NeXT-7B, LLaVA-OneVision-7B, Qwen2.5-VL-7B). All measurements use the
1,000-item VMCBench development set with single-forward letter scoring; depths
are reported as the relative position ℓ/N of layer ℓ in an N-layer decoder.
Token positions are partitioned into the image stream S_img (image-token
positions) and the text stream S_txt (all other positions).

**Per-stream computation.** For each layer ℓ we measure how strongly each
stream is transformed,
u_S(ℓ) = E_x [ mean_{i∈S} ‖h_ℓ(i) − h_{ℓ−1}(i)‖ / ‖h_{ℓ−1}(i)‖ ],
the mean relative residual-stream update over positions i ∈ S (Fig. X, color
strips; each strip is normalized by its own per-layer mean).

**Fusion depth.** Accuracy alone cannot localize fusion, so we use an
attention-knockout sweep. Let A(d) denote letter accuracy when text-query →
image-key attention is blocked in all layers [0, d), and Ã(d) the same
quantity with a mismatched image (a control for priors and formatting); d = 0
denotes the unblocked model. With the *usable image signal* defined as
R₀ = A(0) − Ã(0), we define the retained signal r(d) =
[A(d) − Ã(d)] / R₀ and the *fusion-depth distribution*
φ(d) ∝ max(0, r(d−1) − r(d)),
i.e., the fraction of usable image signal that becomes unrecoverable when layer
d is added to the blocked prefix. φ summarizes *where the answer-relevant
cross-modal transfer happens*; we report its median, interquartile range, and
[q10, q90] (Fig. X, box-and-whisker). Conclusions are unchanged under an
NLL-based variant of r(d) and are stable across question categories
(bootstrapped 95% CIs on the median of ±0.03–0.08 for n_causal ≥ 166).

**Finding 1: fusion is mid-stack and architecture-independent.** Across all
six native models the median fusion depth lies at 0.48–0.62 of the stack,
with tight interquartile ranges; the encoder-based Qwen2.5-VL (0.54) and
LLaVA-OneVision (0.54) are statistically indistinguishable from this band.
The only early-fusing model is LLaVA-1.5 (median 0.28, q10 = 0.03), and within
its own recipe family the fusion interval migrates back toward mid-stack as
training data and resolution scale (LLaVA-NeXT 0.28→ wider IQR; OneVision
0.54), indicating that early fusion is a property of a small-scale frozen-
encoder recipe rather than of having an encoder.

**Finding 2: the pre-fusion layers split the work by stream, not by layer.**
In encoder-based models the image strip is cold before the fusion interval:
early layers transform text while image tokens pass through nearly unchanged.
Native models instead perform substantial early image-stream computation —
the in-LLM analogue of a vision encoder. NEO localizes this work in its
inserted pre-Buffer (image-stream work concentrated before the boundary marker
in Fig. X, text nearly idle there); Gemma-4 and SAIL interleave image and text
computation in the same early layers. Crucially, text processing is never
displaced: every model except NEO shows ordinary early text-stream computation,
and NEO's text pipeline is simply shifted behind its inserted pre-Buffer
(its post-buffer layers are the original 28/36-layer text LLM).

**Causal validation.** Two interventions confirm that the strips reflect
necessary computation rather than incidental drift. (i) *Stream freezing*:
holding the image stream at its layer-0 value through the first 20% of layers
leaves the usable image signal intact in encoder-based models (R₀ retention
1.00 for Qwen2.5-VL, 0.80 for LLaVA-1.5) but destroys it in native models
(0.00–0.27); freezing the text stream instead destroys the signal in every
model (≤0.10) except NEO (0.58), whose pre-Buffer performs little text
computation. (ii) A complementary *suffix-blocking* sweep (blocking layers
[d, N)) shows that no model has transferred more than 20% of its usable image
signal before depth 0.22, ruling out early fusion masked by redundancy.

**Takeaway.** A vision encoder relieves the LLM of visual feature-building —
native models must spend early-layer capacity on it — but it does not move
fusion: in both families, answer-relevant cross-modal transfer is anchored to
the middle of the language model, after the question representation has been
built and regardless of whether the image arrived pre-encoded.

---

## Figure caption

**Figure X. Division of labor over depth in ten VLMs (six native, four
encoder-based).** Each row shows one model over relative depth ℓ/N. The red
strip gives the per-layer image-stream work u_img(ℓ) and the purple strip the
text-stream work u_txt(ℓ) (mean relative residual update over the stream's
positions; each strip normalized by its own per-layer mean). The black
box-and-whisker marks the fusion-depth distribution φ(d) obtained by
cumulative text→image attention knockout: whiskers span [q10, q90], the box
[q25, q75], and the dot the median depth at which usable image signal
(R₀ = accuracy gain from the correct vs. a mismatched image) becomes
unrecoverable. Dashed verticals with triangles mark NEO's pre-Buffer boundary.
Native models perform encoder-like image-stream computation in early layers
(hot red before the boundary), encoder-based models leave the image stream
nearly untouched, and text-stream computation occupies the early layers of
every model except NEO, whose text pipeline is shifted behind its inserted
pre-Buffer. In all models but LLaVA-1.5/NeXT, the fusion interval lies in the
middle of the stack.
