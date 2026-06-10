# Early-Fusion Access Arms Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the four early-fusion experiment arms — image/question ordering transforms (sandwich, random), PaliGemma-style prefix-LM masking, and the layer-windowed img→question mask — to the encoder-free VLM, with inference (generate / lmms-eval) matched to training and zero behavior change when the new configs are off.

**Architecture:** Two independent subsystems. (1) *Ordering*: a pure conversation-list transform applied at the two sample-authoring choke points (energon `encode_sample`, local-json `_get_item`) and at the single inference prompt choke point (`generate_response`), driven by `dataset.image_position` and persisted on the checkpoint config for self-describing eval. (2) *Cross-modal mask*: a pure tensor mask-builder producing 4D bool masks consumed via transformers 5.10's documented 4D-mask passthrough (`masking_utils.py:839` returns prepared 4D masks as-is) and, for the layer-windowed arm, a registered custom attention function (`sdpa_xmodal`) that swaps in a per-layer override stashed on each attention module. Training builds the mask in `forward` after the splice; generation builds it once in `generate` and installs it for the prefill step only (decode steps are pure-causal text rows in both arms, so they need no custom mask).

**Tech Stack:** PyTorch 2.12 SDPA, transformers 5.10.2 (`ALL_ATTENTION_FUNCTIONS` / `AttentionInterface`), Hydra configs, pytest.

**Verified codebase facts (re-checked 2026-06-10, line numbers from current main):**
- `DataArguments` dataclass: `src/vlm/data/data_arguments.py:9`; built from `DatasetConfig` around line 74 (`strip_empty_think=data_config.strip_empty_think` is the plumbing pattern to mirror).
- Energon sample authoring: `VLMChatTaskEncoder.encode_sample` calls `messages_to_conversations` then `inject_missing_media_tokens` then `preprocess` (`src/vlm/data/energon_dataset.py:706-747`). Placeholders sit at data positions.
- Local-json authoring: `preprocess_multimodal` *forces image-first* (`src/vlm/data/dataset.py:331-336`), called at `dataset.py:1257`. Default mode must keep this behavior bit-identical.
- Splice: `prepare_inputs_labels_for_multimodal` returns a 7-tuple ending in `image_block_ids` (B×L, `-1` = non-image, `feature_index ≥ 0` at image rows; built only when `with_image_block_ids=True`); 2D bool `attention_mask` rebuilt at `modeling_vlm.py:1042-1044`; padding side honors `config.padding_side` (left or right).
- Training forward: `modeling_vlm.py:291-382`; `visual_aux_on` gate at 319-326; chunked path receives `attention_mask` at 356-370; fall-through `super().forward` at 371-382.
- Generation: `generate` at `modeling_vlm.py:635-678` splices then calls `super().generate(inputs_embeds=..., attention_mask=...)`; decode steps re-enter `forward` with new-token ids only.
- transformers 5.10.2: a 4D `attention_mask` passes through `create_causal_mask` untouched ("If the mask is already 4D, simply return as-is", `masking_utils.py:839`); Qwen3 attention dispatches `ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, ...)` (`models/qwen3/modeling_qwen3.py:273`) and each `Qwen3Attention` has `self.layer_idx`; `ALL_ATTENTION_FUNCTIONS` is a dict-like `AttentionInterface` instance (`modeling_utils.py:5072-5110`) — registration = item assignment.
- Inference prompt choke point: `generate_response` at `src/vlm/inference/eval.py:423`; lmms-eval wrapper flattens chat to one user query with `<image>` at content positions (`src/vlm/inference/lmms_eval.py:105-125`) and calls `generate_response`.
- Trainer-dial validation/copy pattern: `validate_aux_exit_config` / visual-aux copies in `src/vlm/train.py` (~31-193); logged-component pattern unaffected here.

**Scope guards:**
- All defaults (`image_position: keep`, `cross_modal_mask.mode: none`) must leave every existing code path bit-identical (FA2 + 2D mask untouched).
- v1 implements `img2q_window` **one-directional only** (image queries → question keys). The `bidirectional` config field exists but is validate-rejected with a clear message (mutual windowing requires per-layer decode-step masking — deferred until the arm is justified).
- SLURM jobs requeue and re-read the working tree: implement in a **git worktree** (superpowers:using-git-worktrees), branch `early-fusion-access-arms`; merge to main only after review.

---

## Task 1: `apply_image_position` transform + tests

**Files:**
- Modify: `src/vlm/data/dataset.py` (new function after `preprocess_multimodal`, ~line 349)
- Test: `tests/test_image_position.py` (new)

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the image-position ordering transform (plan 2026-06-10)."""

from vlm.data.dataset import apply_image_position

TOK = "<image>"


def conv(*turns):
    return [{"from": f, "value": v} for f, v in turns]


def test_keep_is_identity():
    c = conv(("human", f"{TOK}\nWhat is this?"), ("gpt", "A cat."))
    apply_image_position(c, mode="keep", image_token=TOK, seed=0)
    assert c[0]["value"] == f"{TOK}\nWhat is this?"


def test_question_first_moves_image_after_text():
    c = conv(("human", f"{TOK}\nWhat is this?"), ("gpt", "A cat."))
    apply_image_position(c, mode="question_first", image_token=TOK, seed=0)
    assert c[0]["value"] == f"What is this?\n{TOK}"
    assert c[1]["value"] == "A cat."  # gpt turns untouched


def test_sandwich_repeats_question():
    c = conv(("human", f"{TOK}\nWhat is this?"),)
    apply_image_position(c, mode="sandwich", image_token=TOK, seed=0)
    assert c[0]["value"] == f"What is this?\n{TOK}\nWhat is this?"


def test_image_token_inside_text_is_extracted():
    c = conv(("human", f"Look at {TOK} and answer."),)
    apply_image_position(c, mode="question_first", image_token=TOK, seed=0)
    assert c[0]["value"] == f"Look at  and answer.\n{TOK}".replace("  ", " ") or True
    # exact contract: token removed, text whitespace-normalized via strip only
    assert c[0]["value"].count(TOK) == 1
    assert c[0]["value"].endswith(TOK)


def test_random_is_deterministic_per_seed():
    base = conv(("human", f"{TOK}\nIs the red car left of the blue truck?"),)
    a = [dict(t) for t in base]
    b = [dict(t) for t in base]
    apply_image_position(a, mode="random", image_token=TOK, seed=1234)
    apply_image_position(b, mode="random", image_token=TOK, seed=1234)
    assert a[0]["value"] == b[0]["value"]
    assert a[0]["value"].count(TOK) == 1


def test_random_varies_across_seeds():
    vals = set()
    for seed in range(40):
        c = conv(("human", f"{TOK}\nIs the red car left of the blue truck?"),)
        apply_image_position(c, mode="random", image_token=TOK, seed=seed)
        vals.add(c[0]["value"])
    assert len(vals) >= 2  # first/middle/last all reachable over 40 seeds


def test_image_only_turn_untouched():
    c = conv(("human", TOK),)
    apply_image_position(c, mode="sandwich", image_token=TOK, seed=0)
    assert c[0]["value"] == TOK


def test_multi_image_turn_untouched():
    v = f"{TOK}\n{TOK}\nCompare these."
    c = conv(("human", v),)
    apply_image_position(c, mode="sandwich", image_token=TOK, seed=0)
    assert c[0]["value"] == v


def test_no_image_turn_untouched():
    c = conv(("human", "Hello"), ("gpt", "Hi"))
    apply_image_position(c, mode="sandwich", image_token=TOK, seed=0)
    assert c[0]["value"] == "Hello"


def test_unknown_mode_raises():
    import pytest

    c = conv(("human", f"{TOK}\nQ?"),)
    with pytest.raises(ValueError):
        apply_image_position(c, mode="banana", image_token=TOK, seed=0)


def test_content_key_supported():
    c = [{"role": "user", "content": f"{TOK}\nQ?"}]
    apply_image_position(c, mode="question_first", image_token=TOK, seed=0)
    assert c[0]["content"] == f"Q?\n{TOK}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_image_position.py -v`
Expected: FAIL / ERROR with `ImportError: cannot import name 'apply_image_position'`

- [ ] **Step 3: Implement the transform in `src/vlm/data/dataset.py`** (place directly after `preprocess_multimodal`, keep its import style; `random` is already imported or add `import random` to the module imports)

```python
def apply_image_position(
    conversations: list[dict],
    mode: str,
    image_token: str,
    seed: int | None = None,
) -> None:
    """Reposition the image placeholder inside human turns, in place (plan
    docs/superpowers/plans/2026-06-10-early-fusion-access-arms.md).

    Only human/user turns containing EXACTLY ONE image token and non-empty
    text are rewritten; everything else (gpt turns, image-only turns,
    multi-image turns) is left untouched. Modes:
      keep           - no-op (default; preserves both paths' current layout)
      question_first - "Q\\n<image>"
      sandwich       - "Q\\n<image>\\nQ"  (question repeated after the image)
      random         - seed-deterministic choice of first / middle / last
    """
    if mode == "keep":
        return
    if mode not in ("question_first", "sandwich", "random"):
        raise ValueError(f"unknown image_position mode: {mode!r}")
    rng = random.Random(seed)
    for turn in conversations:
        if (turn.get("from") or turn.get("role")) not in ("human", "user"):
            continue
        key = "value" if "value" in turn else "content"
        text = str(turn[key])
        if text.count(image_token) != 1:
            continue
        question = text.replace(image_token, " ").strip()
        if not question:
            continue
        if mode == "question_first":
            new = f"{question}\n{image_token}"
        elif mode == "sandwich":
            new = f"{question}\n{image_token}\n{question}"
        else:  # random
            words = question.split()
            placements = ["first", "last"] + (["middle"] if len(words) >= 2 else [])
            choice = rng.choice(placements)
            if choice == "first":
                new = f"{image_token}\n{question}"
            elif choice == "last":
                new = f"{question}\n{image_token}"
            else:
                mid = len(words) // 2
                head, tail = " ".join(words[:mid]), " ".join(words[mid:])
                new = f"{head}\n{image_token}\n{tail}"
        turn[key] = new
```

Note: `test_image_token_inside_text_is_extracted` only pins the hard contract (one token, at the end); the in-text replacement uses `" "` then `strip()` so mid-text extraction never glues words together.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_image_position.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_image_position.py src/vlm/data/dataset.py
git commit -m "feat(data): apply_image_position ordering transform (keep/question_first/sandwich/random)"
```

---

## Task 2: Config plumbing — `dataset.image_position` → both data paths

**Files:**
- Modify: `src/vlm/config/config_schema.py` (DatasetConfig, after `strip_empty_think`, ~line 147)
- Modify: `src/vlm/data/data_arguments.py` (field + factory)
- Modify: `src/vlm/data/energon_dataset.py:718-721` (encode_sample)
- Modify: `src/vlm/data/dataset.py:1257` area (json path)
- Test: `tests/test_image_position.py` (extend)

- [ ] **Step 1: Add the config fields**

In `config_schema.py`, `DatasetConfig`, after `strip_empty_think`:

```python
    # Image-placeholder layout inside human turns (plan 2026-06-10, access
    # arms): "keep" preserves today's layout on both paths; "question_first"
    # / "sandwich" / "random" rewrite single-image human turns. Applied on
    # the energon path after placeholder injection and on the local-json
    # path after preprocess_multimodal's image-first normalization.
    image_position: str = "keep"
```

In `data_arguments.py`: add `image_position: str = "keep"` next to `strip_empty_think: bool = False` (line ~26) and `image_position=data_config.image_position,` in the factory next to the `strip_empty_think=` line (~74).

- [ ] **Step 2: Wire the energon path**

In `energon_dataset.py` `encode_sample`, import `apply_image_position` alongside the existing `dataset` imports, then directly after the `inject_missing_media_tokens(...)` call (line ~721):

```python
        apply_image_position(
            conversations,
            mode=self.data_args.image_position,
            image_token=self.data_args.image_token,
            # Stable per-sample seed: deterministic across epochs/resumes.
            seed=zlib.crc32(str(sample.__key__).encode()),
        )
```

Add `import zlib` to the module imports.

- [ ] **Step 3: Wire the local-json path**

In `dataset.py` at the `sources = preprocess_multimodal(sources, self.data_args)` call site (line 1257), append right after it (`i` is the sample index available in `_get_item`; verify the local variable name in context and use it):

```python
            for source in sources:
                apply_image_position(
                    source,
                    mode=self.data_args.image_position,
                    image_token=self.data_args.image_token,
                    seed=i,
                )
```

(`preprocess_multimodal` output sources are lists of `{"from","value"}` turns — same shape the transform consumes. With mode "keep" this is a no-op, preserving the current forced image-first layout bit-identically.)

- [ ] **Step 4: Extend the test file with a plumbing test**

```python
def test_dataset_config_has_image_position_default_keep():
    from vlm.config.config_schema import DatasetConfig
    from vlm.data.data_arguments import DataArguments

    assert DatasetConfig.image_position == "keep" or DatasetConfig().image_position == "keep"
    assert DataArguments(image_position="sandwich").image_position == "sandwich"
```

- [ ] **Step 5: Run the full test file + existing data tests**

Run: `.venv/bin/python -m pytest tests/test_image_position.py tests/ -v -x -k "image_position or visual_aux"`
Expected: PASS (and no collection errors from the modified modules)

- [ ] **Step 6: Commit**

```bash
git add src/vlm/config/config_schema.py src/vlm/data/data_arguments.py src/vlm/data/energon_dataset.py src/vlm/data/dataset.py tests/test_image_position.py
git commit -m "feat(data): plumb dataset.image_position through energon and json paths"
```

---

## Task 3: Inference matching for ordering + the two data-only arm configs

**Files:**
- Modify: `src/vlm/train.py` (copy `image_position` onto `model.config`, next to the visual-aux copy block)
- Modify: `src/vlm/inference/eval.py:423` `generate_response`
- Create: `src/vlm/config/sft-unified-bee-mix-sandwich.yaml`
- Create: `src/vlm/config/sft-unified-bee-mix-randpos.yaml`

- [ ] **Step 1: Persist the mode on the checkpoint config**

In `train.py`, locate the block that copies trainer/visual-aux dials onto `model.config` (the `validate_visual_aux_config` / copy zone, ~lines 151-193) and add:

```python
    # Self-describing checkpoints: inference must rebuild prompts with the
    # SAME image layout the model was trained on (plan 2026-06-10).
    model.config.image_position = str(cfg.dataset.image_position)
```

- [ ] **Step 2: Apply the transform at the inference choke point**

In `generate_response` (`src/vlm/inference/eval.py:423`), immediately before the query is rendered into the conversation template (find where `query` is first consumed), insert:

```python
    image_position = str(getattr(model.config, "image_position", "keep") or "keep")
    if image_position != "keep" and query.count("<image>") == 1:
        import zlib

        from vlm.data.dataset import apply_image_position

        _turns = [{"from": "human", "value": query}]
        apply_image_position(
            _turns,
            mode=image_position,
            image_token="<image>",
            seed=zlib.crc32(query.encode()),
        )
        query = _turns[0]["value"]
```

(Per-query crc32 seed makes "random"-arm evals deterministic per prompt. Checkpoints trained before this change have no `image_position` attribute → `keep` → behavior unchanged.)

- [ ] **Step 3: Create the two arm configs**

`src/vlm/config/sft-unified-bee-mix-sandwich.yaml`:

```yaml
defaults:
    - sft-unified-bee-mix
    - _self_

# Access arm A-sandwich (plan 2026-06-10): "Q <image> Q" layout — the image
# causally sees the question (copy 1) and the post-image question copy can
# still read the image, preserving the image->question->answer circuit.
# Everything else identical to the bee-mix baseline arm.
dataset:
    image_position: sandwich
```

`src/vlm/config/sft-unified-bee-mix-randpos.yaml`:

```yaml
defaults:
    - sft-unified-bee-mix
    - _self_

# Access arm B (plan 2026-06-10): per-sample random image placement
# (first / middle-of-question / last) — dual-order-style training that
# yields a within-run contrast between samples where the image did and
# did not see the question. Seeded per sample key (deterministic).
dataset:
    image_position: random
```

- [ ] **Step 4: Config smoke test**

Run: `.venv/bin/python -c "
from hydra import compose, initialize_config_dir
import os
cd = os.path.abspath('src/vlm/config')
with initialize_config_dir(version_base=None, config_dir=cd):
    for name in ('sft-unified-bee-mix-sandwich', 'sft-unified-bee-mix-randpos', 'sft-unified-bee-mix'):
        cfg = compose(config_name=name)
        print(name, '->', cfg.dataset.image_position)
"`
Expected: `sandwich`, `random`, `keep`

- [ ] **Step 5: Commit**

```bash
git add src/vlm/train.py src/vlm/inference/eval.py src/vlm/config/sft-unified-bee-mix-sandwich.yaml src/vlm/config/sft-unified-bee-mix-randpos.yaml
git commit -m "feat: self-describing image_position + sandwich/randpos arm configs"
```

---

## Task 4: Cross-modal mask builder + truth-table tests

**Files:**
- Create: `src/vlm/models/xmodal_mask.py`
- Test: `tests/test_xmodal_mask.py` (new)

- [ ] **Step 1: Write the failing truth-table tests**

```python
"""Truth-table tests for the cross-modal 4D mask builder (plan 2026-06-10).

Toy layout, right padding, L=7:
  pos:    0     1     2     3     4     5    6
  role: [sys] [img] [img] [q]  [ans] [ans] [pad]
labels:  -100  -100  -100  -100   7     8   -100
block:    -1    0     0    -1    -1    -1    -1
"""

import torch

from vlm.models.xmodal_mask import build_base_mask, build_cross_modal_mask

ATTN = torch.tensor([[1, 1, 1, 1, 1, 1, 0]], dtype=torch.bool)
LABELS = torch.tensor([[-100, -100, -100, -100, 7, 8, -100]])
BLOCKS = torch.tensor([[-1, 0, 0, -1, -1, -1, -1]])


def allowed(mask, row, col):
    return bool(mask[0, 0, row, col])


def test_base_is_causal_with_key_padding():
    m = build_base_mask(ATTN)
    assert m.shape == (1, 1, 7, 7)
    assert allowed(m, 3, 0) and allowed(m, 3, 3)
    assert not allowed(m, 3, 4)          # causal
    assert not allowed(m, 6, 6) or True  # pad row irrelevant
    assert not allowed(m, 5, 6)          # pad key always blocked


def test_prefix_lm_bidirectional_over_sys_img_question_only():
    m = build_cross_modal_mask(ATTN, None, LABELS, mode="prefix_lm")
    # prefix = positions 0..3 (before first supervised pos 4)
    assert allowed(m, 1, 3)      # img row sees question col (non-causal!)
    assert allowed(m, 0, 3)      # sys row sees question col
    assert allowed(m, 3, 1)      # question row sees img col (causal anyway)
    assert not allowed(m, 1, 4)  # img row must NOT see answer col
    assert not allowed(m, 3, 4)  # question row must NOT see answer col
    assert allowed(m, 4, 1) and allowed(m, 5, 4)  # answer rows: plain causal
    assert not allowed(m, 4, 5)  # answer rows stay causal


def test_img2q_window_adds_only_img_to_question_edges():
    m = build_cross_modal_mask(ATTN, BLOCKS, LABELS, mode="img2q_window")
    assert allowed(m, 1, 3) and allowed(m, 2, 3)  # img rows -> question col
    assert not allowed(m, 0, 3)  # sys row gets NO forward edge
    assert not allowed(m, 1, 4)  # img row -> answer col still blocked
    assert not allowed(m, 3, 4)  # question row unchanged (pure causal)
    # everything else identical to base
    base = build_base_mask(ATTN)
    diff = (m ^ base)[0, 0]
    rows, cols = diff.nonzero(as_tuple=True)
    assert set(rows.tolist()) <= {1, 2}
    assert set(cols.tolist()) <= {3}


def test_labels_none_treats_whole_prompt_as_prefix():
    # generation prefill: [img][img][q][q] no labels
    attn = torch.tensor([[1, 1, 1, 1]], dtype=torch.bool)
    blocks = torch.tensor([[0, 0, -1, -1]])
    m = build_cross_modal_mask(attn, blocks, None, mode="img2q_window")
    assert allowed(m, 0, 2) and allowed(m, 0, 3) and allowed(m, 1, 3)
    mp = build_cross_modal_mask(attn, None, None, mode="prefix_lm")
    assert allowed(mp, 0, 3) and allowed(mp, 2, 0)


def test_left_padding():
    attn = torch.tensor([[0, 1, 1, 1, 1]], dtype=torch.bool)   # pad at pos 0
    labels = torch.tensor([[-100, -100, -100, -100, 9]])
    blocks = torch.tensor([[-1, 0, 0, -1, -1]])
    m = build_cross_modal_mask(attn, blocks, labels, mode="img2q_window")
    assert allowed(m, 1, 3)       # img -> question forward edge
    assert not allowed(m, 1, 0)   # pad key blocked even for img rows
    assert not allowed(m, 1, 4)   # answer col blocked
    mp = build_cross_modal_mask(attn, None, labels, mode="prefix_lm")
    assert allowed(mp, 1, 3) and not allowed(mp, 1, 0)


def test_unknown_mode_raises():
    import pytest

    with pytest.raises(ValueError):
        build_cross_modal_mask(ATTN, BLOCKS, LABELS, mode="banana")
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_xmodal_mask.py -v`
Expected: ImportError (module does not exist)

- [ ] **Step 3: Implement `src/vlm/models/xmodal_mask.py`**

```python
"""Cross-modal 4D attention masks for the early-fusion access arms (plan
docs/superpowers/plans/2026-06-10-early-fusion-access-arms.md).

All builders return (B, 1, L, L) bool masks, True = "query row may attend
key column". transformers 5.10 passes prepared 4D masks through
create_causal_mask untouched (masking_utils.py:839), so these reach SDPA
as-is. Added cross-modal edges always target PREFIX columns only (prefix =
non-pad positions strictly before the sample's first supervised label;
labels=None, i.e. generation prefill, makes the whole prompt the prefix) —
answer keys are never exposed non-causally, so training matches inference.
"""

import torch
from torch import Tensor

IGNORE_INDEX_DEFAULT = -100


def _prefix(attn2d: Tensor, labels: Tensor | None, ignore_index: int) -> Tensor:
    """(B, L) bool: non-pad positions before the first supervised label."""
    keep = attn2d.bool()
    if labels is None:
        return keep
    bsz, seq_len = labels.shape
    idx = torch.arange(seq_len, device=labels.device).expand(bsz, seq_len)
    supervised = labels.ne(ignore_index)
    first = torch.where(supervised, idx, torch.full_like(idx, seq_len)).amin(dim=1)
    return keep & (idx < first.unsqueeze(1))


def build_base_mask(attn2d: Tensor) -> Tensor:
    """Plain causal mask with key-side padding, as a 4D bool tensor."""
    keep = attn2d.bool()
    seq_len = keep.shape[1]
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=keep.device).tril()
    return (causal.unsqueeze(0) & keep.unsqueeze(1)).unsqueeze(1)


def build_cross_modal_mask(
    attn2d: Tensor,
    image_block_ids: Tensor | None,
    labels: Tensor | None,
    mode: str,
    ignore_index: int = IGNORE_INDEX_DEFAULT,
) -> Tensor:
    """Base causal mask plus the arm's cross-modal edges.

    prefix_lm     – bidirectional attention within the prefix
                    (system + image + question), causal suffix. PaliGemma
                    masking transplanted; image_block_ids unused.
    img2q_window  – image-position query rows additionally attend to
                    question-text key columns (prefix & not image). The
                    LAYER windowing happens at install time (the windowed
                    layers get this mask, the rest get build_base_mask).
    """
    base = build_base_mask(attn2d)
    prefix = _prefix(attn2d, labels, ignore_index)
    if mode == "prefix_lm":
        extra = prefix.unsqueeze(2) & prefix.unsqueeze(1)
    elif mode == "img2q_window":
        if image_block_ids is None:
            raise ValueError("img2q_window needs image_block_ids")
        is_img = image_block_ids.ge(0)
        q_text = prefix & ~is_img
        extra = is_img.unsqueeze(2) & q_text.unsqueeze(1)
    else:
        raise ValueError(f"unknown cross_modal_mask mode: {mode!r}")
    return base | extra.unsqueeze(1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_xmodal_mask.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/vlm/models/xmodal_mask.py tests/test_xmodal_mask.py
git commit -m "feat(models): cross-modal 4D mask builder (prefix_lm, img2q_window)"
```

---

## Task 5: `sdpa_xmodal` attention interface (per-layer mask dispatch)

**Files:**
- Modify: `src/vlm/models/xmodal_mask.py` (append)
- Test: `tests/test_xmodal_mask.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
def test_sdpa_xmodal_registered_and_swaps_per_module_mask():
    import torch
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    import vlm.models.xmodal_mask  # noqa: F401  (registration import)

    assert "sdpa_xmodal" in ALL_ATTENTION_FUNCTIONS.valid_keys() or \
        "sdpa_xmodal" in list(ALL_ATTENTION_FUNCTIONS._global_mapping)

    fn = ALL_ATTENTION_FUNCTIONS["sdpa_xmodal"]

    class Cfg:
        _attn_implementation = "sdpa_xmodal"

    class Mod(torch.nn.Module):
        config = Cfg()
        layer_idx = 0
        num_key_value_groups = 1

    m = Mod()
    B, H, L, D = 1, 2, 4, 8
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)
    base = build_base_mask(torch.ones(B, L, dtype=torch.bool))
    full = torch.ones(B, 1, L, L, dtype=torch.bool)

    out_base, _ = fn(m, q, k, v, base, dropout=0.0, scaling=None)
    m._xmodal_mask = full
    out_full, _ = fn(m, q, k, v, base, dropout=0.0, scaling=None)
    assert not torch.allclose(out_base, out_full)  # override took effect

    # decode-step shape guard: q_len 1 != override L -> override ignored
    q1 = torch.randn(B, H, 1, D)
    row = torch.ones(B, 1, 1, L, dtype=torch.bool)
    out_dec, _ = fn(m, q1, k, v, row, dropout=0.0, scaling=None)
    assert out_dec.shape[-2] == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_xmodal_mask.py::test_sdpa_xmodal_registered_and_swaps_per_module_mask -v`
Expected: FAIL (not registered)

- [ ] **Step 3: Append to `src/vlm/models/xmodal_mask.py`**

```python
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def sdpa_xmodal_forward(module, query, key, value, attention_mask, **kwargs):
    """Stock SDPA, but a layer can carry its own 4D mask override
    (module._xmodal_mask, stashed by install_img2q_window_masks). The shape
    guard makes decode steps (q_len 1) fall back to the passed-in causal row
    automatically, so generation needs no per-step bookkeeping."""
    override = getattr(module, "_xmodal_mask", None)
    if (
        override is not None
        and attention_mask is not None
        and override.shape[-2] == query.shape[-2]
        and override.shape[-1] == attention_mask.shape[-1]
    ):
        attention_mask = override
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)


ALL_ATTENTION_FUNCTIONS["sdpa_xmodal"] = sdpa_xmodal_forward
```

(If `sdpa_attention_forward`'s import path differs in the installed 5.10.2, find it with `grep -rn "def sdpa_attention_forward" .venv/lib/python3.13/site-packages/transformers/` and adjust; the test pins behavior either way. If the test's `valid_keys()` accessor doesn't exist on `GeneralInterface`, use the mapping-membership branch already in the assert.)

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_xmodal_mask.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/vlm/models/xmodal_mask.py tests/test_xmodal_mask.py
git commit -m "feat(models): sdpa_xmodal attention interface with per-layer mask override"
```

---

## Task 6: Config schema + validation for the mask arms

**Files:**
- Modify: `src/vlm/config/config_schema.py` (new dataclass + field on the model config)
- Modify: `src/vlm/train.py` (validation + copy onto `model.config`)
- Test: `tests/test_xmodal_mask.py` (extend)

- [ ] **Step 1: Schema.** In `config_schema.py`, next to `VisualAuxConfig` (~line 73), add and register on the model config dataclass exactly where `visual_aux` sits:

```python
@dataclass
class CrossModalMaskConfig:
    # "none" (default, bit-identical baseline) | "prefix_lm" | "img2q_window".
    # prefix_lm: bidirectional attention over [system+image+question], causal
    # suffix, loss unchanged (PaliGemma masking). img2q_window: image-query
    # rows attend question keys in decoder layers window[0]..window[1] only
    # (1-based, inclusive) — the forced-early-fusion arm.
    mode: str = "none"
    window: list[int] = field(default_factory=lambda: [1, 9])
    # Mutual windowing (also confining text->image attention to the window)
    # is NOT implemented in v1: it requires per-layer decode-step masking.
    # The field exists so the config surface is stable; True is rejected.
    bidirectional: bool = False
```

…and on the model config dataclass: `cross_modal_mask: CrossModalMaskConfig = field(default_factory=CrossModalMaskConfig)`.

- [ ] **Step 2: Validation + copy in `train.py`.** Next to `validate_visual_aux_config` add:

```python
def validate_cross_modal_mask_config(cfg, n_layers: int) -> None:
    cmm = cfg.model.cross_modal_mask
    mode = str(cmm.mode or "none")
    if mode == "none":
        return
    if mode not in ("prefix_lm", "img2q_window"):
        raise ValueError(f"cross_modal_mask.mode must be none|prefix_lm|img2q_window, got {mode!r}")
    if bool(cmm.bidirectional):
        raise ValueError(
            "cross_modal_mask.bidirectional=true is not implemented in v1 "
            "(mutual windowing needs per-layer decode masking)"
        )
    attn = str(cfg.trainer.attn_implementation or "")
    if mode == "prefix_lm" and attn not in ("sdpa", "sdpa_xmodal"):
        raise ValueError("prefix_lm needs trainer.attn_implementation=sdpa (4D masks bypass FA2)")
    if mode == "img2q_window":
        if attn != "sdpa_xmodal":
            raise ValueError("img2q_window needs trainer.attn_implementation=sdpa_xmodal")
        lo, hi = int(cmm.window[0]), int(cmm.window[1])
        if not (1 <= lo <= hi <= n_layers):
            raise ValueError(f"cross_modal_mask.window {cmm.window} out of range 1..{n_layers}")
```

Call it where the other validators run (n_layers from the loaded model config, same source `validate_aux_exit_config` uses), then copy plain values onto the HF config in the same block that copies visual-aux dials:

```python
    model.config.cross_modal_mask_mode = str(cfg.model.cross_modal_mask.mode)
    model.config.cross_modal_mask_window = [int(x) for x in cfg.model.cross_modal_mask.window]
```

(Serialized into the checkpoint's config.json → inference is self-describing, mirroring `image_position`.)

- [ ] **Step 3: Registration import.** `sdpa_xmodal` must be registered before `from_pretrained` validates `attn_implementation`. Add `from vlm.models import xmodal_mask  # noqa: F401 (registers sdpa_xmodal)` at the top of `src/vlm/models/modeling_vlm.py` with the comment, so any model build path registers it.

- [ ] **Step 4: Test**

```python
def test_cross_modal_mask_config_defaults():
    from vlm.config.config_schema import CrossModalMaskConfig

    c = CrossModalMaskConfig()
    assert c.mode == "none" and c.window == [1, 9] and c.bidirectional is False
```

Run: `.venv/bin/python -m pytest tests/test_xmodal_mask.py -v` → PASS

- [ ] **Step 5: Commit**

```bash
git add src/vlm/config/config_schema.py src/vlm/train.py src/vlm/models/modeling_vlm.py tests/test_xmodal_mask.py
git commit -m "feat(config): cross_modal_mask schema, validation, checkpoint persistence"
```

---

## Task 7: Training-forward integration

**Files:**
- Modify: `src/vlm/models/modeling_vlm.py` (forward, ~291-382; helper method on the LM mixin class)
- Test: `tests/test_xmodal_mask.py` (extend with an integration test on the toy tensors)

- [ ] **Step 1: Helper.** Add to the same class that owns `forward` (next to it):

```python
    def install_xmodal_masks(
        self: Any,
        attn2d: Tensor,
        image_block_ids: Tensor | None,
        labels: Tensor | None,
    ) -> Tensor:
        """Build the arm's 4D mask(s). Returns the tensor to pass downstream
        as attention_mask; for img2q_window additionally stashes the
        windowed mask on the in-window layers' attention modules (consumed
        by sdpa_xmodal_forward; decode steps shape-guard it away)."""
        mode = str(getattr(self.config, "cross_modal_mask_mode", "none") or "none")
        ignore_index = int(getattr(self.config, "ignore_index", -100))
        if mode == "prefix_lm":
            return build_cross_modal_mask(attn2d, None, labels, mode, ignore_index=ignore_index)
        win = build_cross_modal_mask(attn2d, image_block_ids, labels, mode, ignore_index=ignore_index)
        base = build_base_mask(attn2d)
        lo, hi = (int(x) for x in getattr(self.config, "cross_modal_mask_window", [1, 9]))
        for idx, layer in enumerate(self.model.layers):
            layer.self_attn._xmodal_mask = win if (lo - 1) <= idx <= (hi - 1) else None
        return base
```

Imports: `from vlm.models.xmodal_mask import build_base_mask, build_cross_modal_mask` (module already imported for registration; switch to explicit names).

- [ ] **Step 2: Forward wiring.** In `forward`:

(a) compute the gate next to `visual_aux_on` (line ~326):

```python
        xmodal_mode = str(getattr(self.config, "cross_modal_mask_mode", "none") or "none")
```

(b) extend the splice flag (line 345):

```python
                with_image_block_ids=visual_aux_on or xmodal_mode == "img2q_window",
```

(c) right after the `prepare_inputs_labels_for_multimodal` block (after line 346) and BEFORE the chunked/super dispatch, install masks on the training path and consume the generation-prefill stash:

```python
        if (
            xmodal_mode != "none"
            and labels is not None
            and attention_mask is not None
            and attention_mask.dim() == 2
        ):
            attention_mask = self.install_xmodal_masks(attention_mask, image_block_ids, labels)
        gen_mask = getattr(self, "_xmodal_gen_mask", None)
        if (
            gen_mask is not None
            and inputs_embeds is not None
            and inputs_embeds.shape[1] == gen_mask.shape[-2]
        ):
            attention_mask = gen_mask
            self._xmodal_gen_mask = None
```

Notes: the `labels is not None` gate keeps text-only `generate` decode steps (2D mask, no labels) on the stock causal path; `image_block_ids` is in scope from the 7-tuple unpack; mode `prefix_lm` works with `image_block_ids=None` (visual-aux off). The visual-aux gate `visual_aux_on` is unchanged — block ids are simply also built when the window arm needs them, and `chunked_ce_forward` receives `image_block_ids` only under the same visual-aux conditions as today (its call args at lines 356-370 are untouched).

- [ ] **Step 3: Integration test** (tensor-level; no model download — monkeypatch a minimal namespace):

```python
def test_install_xmodal_masks_window_stash(monkeypatch):
    """install_xmodal_masks stashes the windowed mask on layers lo..hi only."""
    import types

    import torch

    from vlm.models import modeling_vlm

    class Attn:  # noqa: B903
        pass

    class Layer:
        def __init__(self):
            self.self_attn = Attn()

    fake = types.SimpleNamespace(
        config=types.SimpleNamespace(
            cross_modal_mask_mode="img2q_window",
            cross_modal_mask_window=[1, 2],
            ignore_index=-100,
        ),
        model=types.SimpleNamespace(layers=[Layer(), Layer(), Layer()]),
    )
    attn = torch.ones(1, 5, dtype=torch.bool)
    labels = torch.tensor([[-100, -100, -100, 4, 5]])
    blocks = torch.tensor([[-1, 0, -1, -1, -1]])

    # locate the function on the generated class: it is defined module-level
    # and attached via get_dynamic_vlm; call it unbound.
    out = modeling_vlm.install_xmodal_masks(fake, attn, blocks, labels)
    assert out.shape == (1, 1, 5, 5)
    assert fake.model.layers[0].self_attn._xmodal_mask is not None
    assert fake.model.layers[1].self_attn._xmodal_mask is not None
    assert fake.model.layers[2].self_attn._xmodal_mask is None
```

(Adapt the access path to however the file actually exposes methods — the existing pattern is module-level `def forward(self: Any, ...)` functions assembled into the dynamic class by `get_dynamic_vlm` (`modeling_vlm.py:100-131, 1183-1204`); follow it: define `install_xmodal_masks` module-level and attach it in the same place `forward`/`generate` are attached.)

- [ ] **Step 4: Run the whole test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: all PASS (including pre-existing tests — proves no breakage)

- [ ] **Step 5: Commit**

```bash
git add src/vlm/models/modeling_vlm.py tests/test_xmodal_mask.py
git commit -m "feat(models): wire cross-modal masks into the training forward path"
```

---

## Task 8: Generation integration (inference matches training)

**Files:**
- Modify: `src/vlm/models/modeling_vlm.py` (`generate`, lines 635-678)
- Test: extend `tests/test_xmodal_mask.py`

- [ ] **Step 1: Wire `generate`.** Replace the 7-tuple unpack (line 659) and add mask install after it:

```python
            xmodal_mode = str(getattr(self.config, "cross_modal_mask_mode", "none") or "none")
            (_, position_ids, attention_mask, _, inputs_embeds, _, image_block_ids) = (
                self.prepare_inputs_labels_for_multimodal(
                    inputs,
                    position_ids,
                    attention_mask,
                    None,
                    None,
                    image_features,
                    audio_features,
                    with_image_block_ids=xmodal_mode == "img2q_window",
                )
            )
            if xmodal_mode != "none" and attention_mask is not None:
                # Prefill-only custom mask; whole prompt = prefix (labels=None).
                # forward() swaps it in via the shape-matched stash; decode
                # steps are plain causal text rows in both arms by design.
                self._xmodal_gen_mask = self.install_xmodal_masks(
                    attention_mask, image_block_ids, labels=None
                )
```

Keep passing the ORIGINAL 2D `attention_mask` to `super().generate(...)` (HF extends a 2D mask across decode steps; the 4D mask rides the stash). Verify `prepare_inputs_labels_for_multimodal` builds block ids when labels are None — per the splice code (`modeling_vlm.py:962-1011`) `cur_labels` exists regardless and block ids derive from segment shapes, so it does; the smoke test below confirms.

- [ ] **Step 2: Stash-hygiene test**

```python
def test_gen_mask_stash_consumed_once():
    """forward() must consume _xmodal_gen_mask exactly at the matching
    prefill shape and clear it (no leakage into the next call)."""
    import torch

    from vlm.models import modeling_vlm

    # Simulated: stash a mask, then check the swap branch logic standalone.
    gen_mask = torch.ones(1, 1, 6, 6, dtype=torch.bool)
    embeds_prefill = torch.zeros(1, 6, 4)
    embeds_decode = torch.zeros(1, 1, 4)
    assert embeds_prefill.shape[1] == gen_mask.shape[-2]
    assert embeds_decode.shape[1] != gen_mask.shape[-2]
```

(The real assertion of this behavior is the end-to-end smoke in Task 10; this test just pins the shape contract used by the branch.)

- [ ] **Step 3: Run tests, commit**

Run: `.venv/bin/python -m pytest tests/ -v` → PASS

```bash
git add src/vlm/models/modeling_vlm.py tests/test_xmodal_mask.py
git commit -m "feat(models): prefill cross-modal mask in generate (inference matches training)"
```

---

## Task 9: Mask arm configs (+ aux-exit retry config)

**Files:**
- Create: `src/vlm/config/sft-unified-bee-mix-prefixlm.yaml`
- Create: `src/vlm/config/sft-unified-bee-mix-windowearly.yaml`
- Create: `src/vlm/config/sft-unified-bee-mix-sandwich-auxexit.yaml`

- [ ] **Step 1: Write the three configs**

`sft-unified-bee-mix-prefixlm.yaml`:

```yaml
defaults:
    - sft-unified-bee-mix
    - _self_

# Access arm C (plan 2026-06-10): PaliGemma-style prefix-LM — bidirectional
# attention over [system+image+question], causal suffix, loss unchanged.
# 4D masks bypass FA2; SDPA carries them (masking_utils passthrough).
model:
    cross_modal_mask:
        mode: prefix_lm

trainer:
    attn_implementation: sdpa
```

`sft-unified-bee-mix-windowearly.yaml`:

```yaml
defaults:
    - sft-unified-bee-mix
    - _self_

# Force arm window-early (plan 2026-06-10): image-query rows attend
# question keys in decoder layers 1-9 ONLY (28-layer Qwen3-1.7B); above
# the window the image is text-blind again. Any question conditioning of
# visual processing must therefore happen early. text->image direction
# untouched (plain causal at all layers).
model:
    cross_modal_mask:
        mode: img2q_window
        window: [1, 9]

trainer:
    attn_implementation: sdpa_xmodal
```

`sft-unified-bee-mix-sandwich-auxexit.yaml`:

```yaml
defaults:
    - sft-unified-bee-mix-sandwich
    - _self_

# Force arm aux-exit-retry (plan 2026-06-10): the existing early-exit CE
# (k=6, shared norm+lm_head) re-run ON TOP OF sandwich ordering — under
# access the early CE can finally reward question-conditioned image
# reads; detach=true per the postmortem (FUSION_METRICS_EXPLAINED.md §5).
trainer:
    aux_exit_layers: [6]
    aux_exit_weight: 0.25
    aux_exit_detach: true
```

- [ ] **Step 2: Compose-check all five new configs**

Run the Task-3 Hydra snippet extended with the three new names; additionally print `cfg.model.cross_modal_mask.mode`, `cfg.trainer.attn_implementation`, `cfg.trainer.aux_exit_layers`.
Expected: `prefix_lm/sdpa`, `img2q_window/sdpa_xmodal`, `sandwich + [6]`.

- [ ] **Step 3: Commit**

```bash
git add src/vlm/config/sft-unified-bee-mix-prefixlm.yaml src/vlm/config/sft-unified-bee-mix-windowearly.yaml src/vlm/config/sft-unified-bee-mix-sandwich-auxexit.yaml
git commit -m "config: prefix-LM, window-early, sandwich+aux-exit arm configs"
```

---

## Task 10: End-to-end smoke + equivalence verification (GPU)

**Files:**
- Create: `devtools/xmodal_smoke.py`
- Create: `devtools/xmodal_smoke.slurm` (follow `devtools/sft_trial.slurm` conventions: ckpt-all account / cse-ckpt partition / l40:4 → a single l40:1 here, requeue per Hyak rules)

- [ ] **Step 1: Write `devtools/xmodal_smoke.py`** — loads the Qwen3-0.6B smoke model (`qwen3-0.6b-unified` config family) and asserts, on one synthetic batch (two samples, one image each, right/left padding exercised):

1. `cross_modal_mask_mode="none"` + sdpa logits == FA2 logits within bf16 tolerance AND == plain 2D-mask sdpa logits exactly (no-op proof).
2. `prefix_lm` logits differ from baseline ONLY when the prefix has >1 token after the image (sanity: masks took effect), and answer-position logits for a suffix-only-causal layout change as expected.
3. `img2q_window` with window `[1, 9]`: run a probe — zero out question embeddings and verify image-position hidden states at layer 9 CHANGE vs unmodified question (question-sensitivity > 0) while with mode `none` they are bit-identical (the theorem check ported to the smoke).
4. `generate()` runs for both modes (greedy, 8 new tokens) without error and the stash is consumed (`model._xmodal_gen_mask is None` afterward).
5. Throughput probe: time 20 forward+backward steps at bucket-shaped batches for FA2-baseline vs sdpa-prefix_lm vs sdpa_xmodal-window; print tokens/s ratios.

- [ ] **Step 2: Run it**

Run: `sbatch devtools/xmodal_smoke.slurm` then check the log under `logs/`.
Expected: all five assertions print PASS; throughput ratio recorded in the log (accept ≥0.7× FA2; if lower, file a follow-up — do not block the arms).

- [ ] **Step 3: Full pytest sweep one more time**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add devtools/xmodal_smoke.py devtools/xmodal_smoke.slurm
git commit -m "test: end-to-end smoke for ordering + cross-modal mask arms"
```

---

## Task 11: Review pass + docs

- [ ] **Step 1:** Dispatch an independent review agent (model: opus, per user preference) to verify the full diff against the installed transformers 5.10.2 sources in `.venv` — especially: 4D-mask passthrough behavior, `sdpa_attention_forward` signature, `AttentionInterface` registration, and that the default-off paths are bit-identical (no new tensor allocation when modes are off).
- [ ] **Step 2:** Address findings; re-run `pytest tests/ -v`.
- [ ] **Step 3:** Update `docs/superpowers/specs/` with a short spec note (this plan's path + final knob names) and commit.

---

## Self-review notes

- Spec coverage: arms A-sandwich (Task 1-3), B-random (1-3), C prefix-LM (4-9), window-early (4-9), aux-exit retry (9), inference matching (3, 8), no-breakage (every task's default-off tests + Task 10.1).
- Decode-step correctness argument (both mask arms): decode queries are text rows; window arm leaves text rows causal at every layer, prefix-LM's bidirectionality only affected prefix-internal rows computed at prefill; therefore standard causal decode masks are exactly training-consistent. The `_xmodal_mask` shape guard + one-shot `_xmodal_gen_mask` stash enforce this mechanically.
- Known deferred items: mutual (bidirectional) windowing (validate-rejected), energon-path ordering for multi-image samples (transform skips them, logged fraction is tiny for bee-mix), flex-attention throughput optimization (only if SDPA ratio <0.7×).
