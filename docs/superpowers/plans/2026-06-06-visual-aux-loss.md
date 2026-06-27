# Visual Auxiliary Losses (aim_pixel / nepa) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Execution status:** completed 2026-06-06 via subagent-driven development on branch `worktree-visual-aux` (15 commits, GPU gate + aux-exit regression green). Checkboxes left unticked; per-task review records live in the session transcript.

**Goal:** Pluggable next-patch prediction auxiliary losses at image positions (pixel targets = AIM-style, embedding targets = NEPA-style with stop-grad) on the encoder-free unified VLM, λ=0.5, with `objective: none` byte-identical to today.

**Architecture:** A fresh MLP head on the causal wrapper (next to `lm_head`), built in `__init__` from `model.config` fields (audio-connector gating pattern). The splice tracks which spliced positions belong to which image (block ids); `chunked_ce_forward` builds shift-by-one prediction pairs strictly within blocks and adds `λ·L_visual` to the CE loss, stashing components for the existing all-reduce logging. Spec: `docs/superpowers/specs/2026-06-06-visual-aux-loss-design.md`.

**Tech Stack:** PyTorch, transformers 5.10.x (dynamic VLM classes), Hydra structured configs, DeepSpeed ZeRO-2, pytest (CPU units) + `devtools/` GPU harness via srun.

**Critical context for the implementer (verified against source, 2026-06-06):**
- `set_trainable_params` (`src/vlm/train/set_trainable.py:104`) freezes ALL params then re-enables groups; params with no matching prefix fall through to the `language_model` group (`set_trainable.py:61-66`). Without explicit registration the new head would be frozen in connector-only recipes and take the LM lr otherwise.
- `VLMTrainer.log` (`src/vlm/train/vlm_trainer.py:119-128`) reads `comps["ce_final"]` / `comps["ce_aux"]` by hard-coded key — crashes if visual-aux stashes a different key set. Must be generalized to sorted-key iteration (rank-deterministic).
- The splice (`src/vlm/models/modeling_vlm.py:652-876`) strips padding, re-assembles, truncates to `config.max_seq_length`, then **re-pads on `config.padding_side`, which is `"left"` for these models** — block-id rows must mirror BOTH padding branches.
- Fresh build (`src/vlm/vlm.py:188-205`): `VLMConfig(...)` is constructed, then `VLMForCausalLM.from_pretrained(hf_name, config=config, ...)` — the causal `__init__` (`modeling_vlm.py:186-191`) runs with that config, so structural fields must be ON the config object before `from_pretrained`. Missing head weights in the HF backbone checkpoint are randomly initialized by transformers (standard "newly initialized" warning) — that's correct, the head is always fresh.
- Plain attributes set on the config serialize into checkpoint `config.json` and are restored as attributes on reload — proven in-repo by `conversation_version` (written in `train.py:108`, read back by `vlm.inference.resolve_conv_mode`).
- `config.vision_config.hidden_size` on the encoder-free path is `(patch_size*pooling_kernel_size)²*3 = 6912` (set at `vlm.py:133`, "single source of truth") — this is the `aim_pixel` head out_dim AND the row dim of the raw-patch tensors in `images`.
- Live-baseline safety: running cluster jobs relaunch from this working tree on requeue. Every task's `objective="none"` path must be behaviorally identical; the GPU parity test (Task 11) is the gate before any requeue window.
- The working tree has unrelated uncommitted modifications (`config_schema.py`, `data_arguments.py`, `energon_dataset.py` + untracked files). **Commit only the files each task names** — never `git add -A`.

---

### Task 1: Config plumbing (Hydra schema → TrainingArguments)

**Files:**
- Modify: `src/vlm/config/config_schema.py` (ModelConfig ~line 73-79, TrainerConfig ~line 217 after `aux_exit_detach`)
- Modify: `src/vlm/train/training_arguments.py` (fields ~line 250, passthrough ~line 313)

- [ ] **Step 1: Add `VisualAuxConfig` + ModelConfig field in `config_schema.py`**

Insert directly above `@dataclass\nclass ModelConfig:` (line 73):

```python
@dataclass
class VisualAuxConfig:
    """Visual auxiliary prediction loss at image positions (spec:
    docs/superpowers/specs/2026-06-06-visual-aux-loss-design.md). Structural
    dials only — they size the head module, so they live on the model config
    (the loss weight/layer are trainer dials). "none" = no head built,
    bit-identical baseline path."""

    # none | aim_pixel (next-patch z-scored pixel MSE, AIM/AIMv2-style)
    #      | nepa (next-patch connector-embedding cosine, stop-grad target)
    objective: str = "none"
    # Head MLP: depth 1 = single Linear; depth d = (d-1) x [Linear, GELU] + Linear.
    head_depth: int = 2
    # Internal width of the head MLP (input is always the LM hidden size).
    head_hidden: int = 2048
```

and add the field to `ModelConfig`:

```python
@dataclass
class ModelConfig:
    name: str = MISSING
    visual_encoder: VisualEncoderConfig = field(default_factory=VisualEncoderConfig)
    language_model: LanguageModelConfig = field(default_factory=LanguageModelConfig)
    connector: ConnectorConfig = field(default_factory=ConnectorConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    visual_aux: VisualAuxConfig = field(default_factory=VisualAuxConfig)
```

- [ ] **Step 2: Add trainer-side fields to `TrainerConfig`** (after `aux_exit_detach: bool = False`, line 217):

```python
    # Visual-aux loss weight λ (spec 2026-06-06): only read when
    # model.visual_aux.objective != "none". L = L_CE + λ·L_visual.
    visual_aux_weight: float = 0.5
    # null = attach the head to the post-final-norm last hidden state;
    # k = decode layer k's output through the shared final RMSNorm first
    # (aux-exit capture mechanism; valid [1, n_layers-1]).
    visual_aux_layer: int | None = None
    # Optimizer dials for the (always-trainable) head; None falls back to
    # default_lr / language_model weight decay.
    visual_aux_head_lr: float | None = None
    visual_aux_head_wd: float | None = None
```

- [ ] **Step 3: Mirror the four fields on `TrainingArguments`** (`training_arguments.py`, after `aux_exit_detach: bool = False`, line 250):

```python
    # Visual-aux loss (spec 2026-06-06): weight/layer are copied onto
    # model.config by train.py; lr/wd are consumed by optimizer.py only.
    visual_aux_weight: float = 0.5
    visual_aux_layer: int | None = None
    visual_aux_head_lr: float | None = None
    visual_aux_head_wd: float | None = None
```

and pass them through in `get_training_args` (after `aux_exit_detach=config.aux_exit_detach,`, line 313):

```python
        visual_aux_weight=config.visual_aux_weight,
        visual_aux_layer=config.visual_aux_layer,
        visual_aux_head_lr=config.visual_aux_head_lr,
        visual_aux_head_wd=config.visual_aux_head_wd,
```

- [ ] **Step 4: Verify configs compose**

Run (login node, repo root, venv active):
```bash
python -c "
from hydra import compose, initialize_config_dir
import vlm.config as c; c.register_configs()
with initialize_config_dir(config_dir='$PWD/src/vlm/config', version_base=None):
    cfg = compose(config_name='sft-unified')
assert cfg.model.visual_aux.objective == 'none'
assert abs(cfg.trainer.visual_aux_weight - 0.5) < 1e-9
assert cfg.trainer.visual_aux_layer is None
print('compose OK')
"
```
Expected: `compose OK`. (If `vlm.config` does not re-export `register_configs`, use `from vlm.config.config_schema import register_configs`.)

- [ ] **Step 5: Commit**

```bash
git add src/vlm/config/config_schema.py src/vlm/train/training_arguments.py
git commit -m "config: visual-aux schema fields (model.visual_aux + trainer dials)"
```

---

### Task 2: Pure helpers — pair construction + target prep (TDD, CPU)

**Files:**
- Modify: `src/vlm/models/modeling_vlm.py` (module level, directly below `_rms_norm`, line 33)
- Create: `tests/test_visual_aux_pairs.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_visual_aux_pairs.py`:

```python
"""CPU unit tests for the visual-aux pair construction + target prep helpers
(spec: docs/superpowers/specs/2026-06-06-visual-aux-loss-design.md §2-3)."""

import torch

from vlm.models.modeling_vlm import build_visual_aux_pairs, prepare_visual_aux_targets


def _row(spec: list[tuple[int, int]], length: int) -> torch.Tensor:
    """spec = [(block_id, n_positions), ...] laid out left to right; -1 fills."""
    row = torch.full((length,), -1, dtype=torch.long)
    cursor = 0
    for block_id, n in spec:
        row[cursor : cursor + n] = block_id
        cursor += n
    return row


def test_pairs_within_blocks_only():
    # row 0: text(2) | img0 patches(4) | text(1) | img1 patches(3) | pad(2)
    ids = torch.stack(
        [
            _row([(-1, 2), (0, 4), (-1, 1), (1, 3)], 12),
            _row([(-1, 12)], 12),  # text-only row (dummy image consumed zero-width)
        ]
    )
    flat_pos, segments = build_visual_aux_pairs(ids, num_target_rows=[4, 3])
    # img0: positions 2,3,4 predict rows 1,2,3 (position 5 = last patch, no target)
    # img1: positions 7,8 predict rows 1,2
    assert segments == [(0, 3), (1, 2)]
    assert flat_pos.tolist() == [2, 3, 4, 7, 8]


def test_pairs_second_batch_row_offset():
    L = 8
    ids = torch.stack([_row([(-1, L)], L), _row([(0, 5), (-1, 3)], L)])
    flat_pos, segments = build_visual_aux_pairs(ids, num_target_rows=[5])
    assert segments == [(0, 4)]
    assert flat_pos.tolist() == [L + 0, L + 1, L + 2, L + 3]


def test_truncated_block_keeps_valid_pairs():
    # img0 has 6 target rows but only 2 positions survived truncation:
    # both surviving positions still have real next-row targets (rows 1, 2).
    ids = _row([(-1, 1), (0, 2)], 3).unsqueeze(0)
    flat_pos, segments = build_visual_aux_pairs(ids, num_target_rows=[6])
    assert segments == [(0, 2)]
    assert flat_pos.tolist() == [1, 2]


def test_single_patch_and_empty():
    ids = _row([(0, 1), (-1, 4)], 5).unsqueeze(0)
    flat_pos, segments = build_visual_aux_pairs(ids, num_target_rows=[1])
    assert segments == [] and flat_pos.numel() == 0
    flat_pos, segments = build_visual_aux_pairs(
        torch.full((2, 5), -1, dtype=torch.long), num_target_rows=[]
    )
    assert segments == [] and flat_pos.numel() == 0


def test_left_padding_offsets():
    # left-padded row: pad(3) | img0(4) | text(1)
    ids = _row([(-1, 3), (0, 4), (-1, 1)], 8).unsqueeze(0)
    flat_pos, segments = build_visual_aux_pairs(ids, num_target_rows=[4])
    assert segments == [(0, 3)]
    assert flat_pos.tolist() == [3, 4, 5]


def test_aim_pixel_targets_zscore_and_shift():
    torch.manual_seed(0)
    img = torch.randn(4, 12, dtype=torch.float32) * 3.0 + 1.5
    tgt = prepare_visual_aux_targets("aim_pixel", [img], [(0, 3)])
    assert tgt.shape == (3, 12) and tgt.dtype == torch.float32
    # rows are img[1:4], each z-scored with the MAE formula (unbiased var, eps 1e-6)
    ref = img[1:4]
    ref = (ref - ref.mean(-1, keepdim=True)) / (ref.var(-1, keepdim=True) + 1e-6).sqrt()
    assert torch.allclose(tgt, ref, atol=1e-6)
    # shift guard: target row 0 is img[1], NOT img[0]
    assert not torch.allclose(tgt[0], (img[0] - img[0].mean()) / (img[0].var() + 1e-6).sqrt())


def test_nepa_targets_detached_and_normalized():
    e = torch.randn(5, 8, requires_grad=True)
    tgt = prepare_visual_aux_targets("nepa", [e], [(0, 4)])
    assert tgt.requires_grad is False  # stop-grad: MANDATORY (collapse guard)
    assert tgt.shape == (4, 8)
    assert torch.allclose(tgt.norm(dim=-1), torch.ones(4), atol=1e-5)
    ref = torch.nn.functional.normalize(e[1:5].detach().float(), dim=-1)
    assert torch.allclose(tgt, ref, atol=1e-6)


def test_multi_image_target_concat_order():
    a = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    b = torch.arange(8, dtype=torch.float32).reshape(2, 4) + 100
    tgt = prepare_visual_aux_targets("nepa", [a, b], [(0, 2), (1, 1)])
    assert tgt.shape == (3, 4)
    raw = torch.cat([a[1:3], b[1:2]]).float()
    ref = torch.nn.functional.normalize(raw, dim=-1)
    assert torch.allclose(tgt, ref, atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_visual_aux_pairs.py -q
```
Expected: `ImportError: cannot import name 'build_visual_aux_pairs'`.

- [ ] **Step 3: Implement the helpers**

In `src/vlm/models/modeling_vlm.py`, directly below `_rms_norm` (after line 32), add:

```python
def build_visual_aux_pairs(
    image_block_ids: Tensor, num_target_rows: list[int]
) -> tuple[Tensor, list[tuple[int, int]]]:
    """Shift-by-one prediction pairs for the visual-aux loss, strictly within
    image blocks (spec: docs/superpowers/specs/2026-06-06-visual-aux-loss-design.md).

    image_block_ids: (B, L) long; -1 on text/audio/padding; the spliced patch
    positions of image k (flat batch-cursor index into the per-image lists)
    carry k. Splice truncation only ever removes a block's TAIL, so the
    surviving positions of block k are its first m patches and position i of
    the block predicts target row i+1 — valid while i+1 <= N_k - 1.

    Returns (flat_positions, segments): flat_positions indexes rows of the
    (B*L, D) flattened hidden states whose hidden state predicts the NEXT
    patch; segments is [(k, n_pairs)] in the same order — the matching
    targets are rows 1..n_pairs of image k (see prepare_visual_aux_targets).
    Blocks with < 2 patches (incl. zero-width dummy images, which never
    receive a block id) contribute nothing.
    """
    _, seq_len = image_block_ids.shape
    flat_positions: list[Tensor] = []
    segments: list[tuple[int, int]] = []
    for batch_idx in range(image_block_ids.shape[0]):
        row = image_block_ids[batch_idx]
        for k_t in torch.unique(row[row >= 0]).tolist():
            k = int(k_t)
            pos = (row == k).nonzero(as_tuple=True)[0]
            n_pairs = min(int(pos.numel()), num_target_rows[k] - 1)
            if n_pairs <= 0:
                continue
            flat_positions.append(batch_idx * seq_len + pos[:n_pairs])
            segments.append((k, n_pairs))
    if not flat_positions:
        return image_block_ids.new_zeros((0,)), []
    return torch.cat(flat_positions), segments


def prepare_visual_aux_targets(
    objective: str, targets_src: list[Tensor], segments: list[tuple[int, int]]
) -> Tensor:
    """Assemble the (n_pairs_total, dim) fp32 target matrix for the visual-aux
    loss. Per segment (k, n): rows 1..n of targets_src[k] (the shift-by-one
    "next patch" — row 0 is target-only, nothing predicts it).

    aim_pixel: targets are REAL pixels (external ground truth — no stop-grad
    needed, no degenerate zero-loss solution exists); per-patch z-score with
    the MAE formula (mean/unbiased-var over the patch dim, eps 1e-6).
    nepa: targets are the connector outputs; detach() is the SimSiam-style
    stop-grad that prevents representational collapse (NEPA ablation: without
    it the cosine slides to 1 and training collapses), then L2-normalize.
    """
    rows = torch.cat([targets_src[k][1 : 1 + n] for k, n in segments])
    if objective == "aim_pixel":
        rows = rows.float()
        mean = rows.mean(dim=-1, keepdim=True)
        var = rows.var(dim=-1, keepdim=True)
        return (rows - mean) / (var + 1.0e-6).sqrt()
    if objective == "nepa":
        return nn.functional.normalize(rows.detach().float(), dim=-1)
    raise ValueError(f"unknown visual_aux objective: {objective!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_visual_aux_pairs.py -q
```
Expected: `8 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/vlm/models/modeling_vlm.py tests/test_visual_aux_pairs.py
git commit -m "feat: visual-aux pair construction + target prep helpers (TDD)"
```

---

### Task 3: Head module — builder, `__init__` attach, build-site wiring

**Files:**
- Modify: `src/vlm/models/modeling_vlm.py` (causal class: `__init__` line 186-191, class dict ~line where `DynamicCausalVLMClass = type(...)` is assembled — search `"prepare_inputs_for_generation"` in the dict)
- Modify: `src/vlm/vlm.py` (fresh build ~line 188-194; from_pretrained branch ~line 106-121)

- [ ] **Step 1: Add the head builder to the causal class**

In `create_dynamic_causal_vlm_class` (`modeling_vlm.py`), add this function next to the other methods (e.g. directly before the `forward` definition at line 193):

```python
    def _build_visual_aux_head(self: Any, config: Any) -> nn.Sequential | None:
        """Visual-aux prediction head (spec 2026-06-06): a small MLP on trunk
        hidden states at image positions, predicting the NEXT patch's pixels
        (aim_pixel; out = vision_config.hidden_size = patch_dim) or connector
        embedding (nepa; out = hidden_size). None when the objective is off —
        baseline models carry no extra module (audio-connector pattern), and
        old checkpoints (no visual_aux_* keys in config.json) load unchanged."""
        objective = str(getattr(config, "visual_aux_objective", "none") or "none")
        if objective == "none":
            return None
        if objective == "aim_pixel":
            # Encoder-free single source of truth: (model patch px)^2 * 3,
            # set by load_model. train.py rejects encoder-present configs.
            out_dim = int(config.vision_config.hidden_size)
        elif objective == "nepa":
            out_dim = int(config.hidden_size)
        else:
            raise ValueError(
                f"unknown visual_aux_objective {objective!r} (none|aim_pixel|nepa)"
            )
        depth = int(getattr(config, "visual_aux_head_depth", 2) or 2)
        hidden = int(getattr(config, "visual_aux_head_hidden", 0) or config.hidden_size)
        layers: list[nn.Module] = []
        in_dim = int(config.hidden_size)
        for _ in range(depth - 1):
            layers.extend([nn.Linear(in_dim, hidden), nn.GELU()])
            in_dim = hidden
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)
```

- [ ] **Step 2: Attach in the causal `__init__`**

Modify the causal `__init__` (line 186-191) to:

```python
    @override
    def __init__(self: Any, config):  # pyright: ignore
        super(self.__class__, self).__init__(config)
        self.model = pretrain_class(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Visual-aux head (spec 2026-06-06): None unless the config carries an
        # objective — fresh HF-backbone loads leave it randomly initialized
        # ("newly initialized" warning is expected; the head is always fresh).
        self.visual_aux_head = self._build_visual_aux_head(config)
        self.post_init()
        log.info(f"DynamicCausalVLM class {self.__class__.__name__} initialized.")
```

and register `"_build_visual_aux_head": _build_visual_aux_head` in the class-construction dict (the `type("VLMForCausalLM", ...)`-style dict at the bottom of `create_dynamic_causal_vlm_class` — add next to the other method entries).

- [ ] **Step 3: Set the structural fields at the fresh-build site**

In `src/vlm/vlm.py`, after the `config = VLMConfig(...)` construction (line 188-194) and BEFORE `VLMForCausalLM.from_pretrained(...)`, add:

```python
        # Visual-aux structural fields (spec 2026-06-06) must be on the config
        # BEFORE model construction — the causal __init__ builds the head from
        # them. Plain python types; they serialize into checkpoint config.json
        # (conversation_version pattern) so reloads rebuild the head.
        config.visual_aux_objective = str(model_cfg.visual_aux.objective)
        config.visual_aux_head_depth = int(model_cfg.visual_aux.head_depth)
        config.visual_aux_head_hidden = int(model_cfg.visual_aux.head_hidden)
```

- [ ] **Step 4: Fail loud on unsupported retrofit**

In the `trainer_cfg.from_pretrained` branch of `load_model` (after the `attach_audio_feature_extractor(...)` call, line 121), add:

```python
        # Visual-aux retrofit guard: enabling the head on an understanding-only
        # checkpoint needs post-hoc init wiring that is deliberately out of v1
        # scope (spec §7) — fail loud instead of silently training without it.
        if (
            str(model_cfg.visual_aux.objective) != "none"
            and getattr(model, "visual_aux_head", None) is None
        ):
            raise ValueError(
                "model.visual_aux.objective is set but the loaded checkpoint has "
                "no visual_aux_head — retrofit from understanding-only checkpoints "
                "is not supported yet; train from scratch (sft-unified-aimpixel / "
                "sft-unified-nepa) or load a checkpoint trained with the head"
            )
```

- [ ] **Step 5: Quick structural check (CPU, no weights)**

```bash
python -c "
import torch, torch.nn as nn
from types import SimpleNamespace
import importlib
m = importlib.import_module('vlm.models.modeling_vlm')
# exercise the builder logic standalone via a throwaway class
src = None
cfg = SimpleNamespace(visual_aux_objective='aim_pixel', visual_aux_head_depth=2,
                      visual_aux_head_hidden=2048, hidden_size=2048,
                      vision_config=SimpleNamespace(hidden_size=6912))
# the builder is defined inside the factory; smoke it through a tiny shim:
import inspect
fn = [c for c in inspect.getsource(m.create_dynamic_causal_vlm_class).split('def ') if c.startswith('_build_visual_aux_head')]
assert fn, 'builder not found in causal factory'
print('builder present OK')
"
```
Expected: `builder present OK`. (Full functional coverage lands in Task 11's GPU test, which builds the real model.)

- [ ] **Step 6: Commit**

```bash
git add src/vlm/models/modeling_vlm.py src/vlm/vlm.py
git commit -m "feat: visual_aux_head module (config-gated, audio-connector pattern)"
```

---

### Task 4: Splice block-id tracking + forward threading

**Files:**
- Modify: `src/vlm/models/modeling_vlm.py` (`prepare_inputs_labels_for_multimodal` 652-876, `forward` 194-263, `generate` call site 457-467)

- [ ] **Step 1: Extend the splice**

Apply these exact changes to `prepare_inputs_labels_for_multimodal`:

(a) Signature + return type — add a keyword param and a 7th return element:

```python
    def prepare_inputs_labels_for_multimodal(
        self: Any,
        input_ids: Tensor | None = None,
        position_ids: LongTensor | None = None,
        attention_mask: Tensor | None = None,
        past_key_values: list[FloatTensor] | None = None,
        labels: LongTensor | None = None,
        image_features: Tensor | None = None,
        audio_features: list[Tensor] | None = None,
        with_image_block_ids: bool = False,
    ) -> tuple[
        Tensor | None,
        LongTensor | None,
        Tensor | None,
        list[FloatTensor] | None,
        Tensor | None,
        LongTensor | None,
        LongTensor | None,
    ]:
```

(b) Early return (line 669-670) becomes:

```python
        if (image_features is None and audio_features is None) or input_ids.shape[1] == 1:  # pyright: ignore
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None
```

(c) Next to `new_input_embeds = []` / `new_labels = []` (line 719-720), add:

```python
        # Visual-aux block-id tracking (spec 2026-06-06 §3.2): which spliced
        # positions belong to which image (flat batch-cursor index). Built only
        # on request so the baseline assembly loop stays byte-identical.
        new_image_block_ids: list[Tensor] | None = [] if with_image_block_ids else None
```

(d) In the `len(mm_positions) == 0` continue-branch (lines 734-739), before `continue`, add:

```python
                if new_image_block_ids is not None:
                    new_image_block_ids.append(
                        torch.full_like(labels[batch_idx], -1)  # pyright: ignore
                    )
```

(e) In the main per-row assembly: next to `cur_new_labels = []` (line 755) add `cur_new_block_ids = []`; then inside the `for i in range(len(mm_positions) + 1)` loop mirror every labels append —

after `cur_new_labels.append(cur_labels_segments[i])` (line 759):

```python
                if new_image_block_ids is not None:
                    cur_new_block_ids.append(
                        torch.full_like(cur_labels_segments[i], -1)
                    )
```

and inside the `if i < len(mm_positions):` branch, capture the cursor BEFORE it advances — replace lines 762-764 with:

```python
                    features, cursor = modality_features[token_index]
                    feature_index = cursor[0]
                    cur_features = features[feature_index]
                    cursor[0] += 1
```

then after the ignore-index labels append (line 766-773), add:

```python
                    if new_image_block_ids is not None:
                        cur_new_block_ids.append(
                            torch.full(
                                (cur_features.shape[0],),
                                feature_index
                                if token_index == self.config.image_token_index
                                else -1,
                                device=cur_labels.device,
                                dtype=cur_labels.dtype,
                            )
                        )
```

and after `cur_new_labels = torch.cat(cur_new_labels)` (line 779):

```python
            if new_image_block_ids is not None:
                new_image_block_ids.append(torch.cat(cur_new_block_ids))
```

(f) Truncation (lines 786-788) — add inside the `if tokenizer_model_max_length is not None:` block:

```python
            if new_image_block_ids is not None:
                new_image_block_ids = [
                    x[:tokenizer_model_max_length] for x in new_image_block_ids
                ]
```

(g) Padding — next to `new_labels_padded = torch.full(...)` (line 795-800), add:

```python
        image_block_ids_padded = (
            torch.full(
                (batch_size, max_len),
                -1,
                dtype=new_labels[0].dtype,
                device=new_labels[0].device,
            )
            if new_image_block_ids is not None
            else None
        )
```

and inside BOTH padding branches, next to the `new_labels_padded[i, ...] = cur_new_labels` lines (829 and 852), add the matching fill — left branch:

```python
                    if image_block_ids_padded is not None:
                        image_block_ids_padded[i, -cur_len:] = new_image_block_ids[i]
```

right branch:

```python
                    if image_block_ids_padded is not None:
                        image_block_ids_padded[i, :cur_len] = new_image_block_ids[i]
```

(h) Final return (line 876):

```python
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, image_block_ids_padded  # pyright: ignore
```

- [ ] **Step 2: Thread through `forward`**

Replace the splice call in `forward` (lines 222-233) with:

```python
        # Visual-aux gating (spec 2026-06-06): block ids are built only when
        # the head exists, λ > 0, and we are on the training-loss path.
        visual_aux_on = (
            self.training
            and labels is not None
            and getattr(self, "visual_aux_head", None) is not None
            and float(getattr(self.config, "visual_aux_weight", 0.0) or 0.0) > 0.0
        )
        image_block_ids = None
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                image_block_ids,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                image_features,
                audio_features,
                with_image_block_ids=visual_aux_on,
            )
```

and extend the chunked call (lines 242-251) with the three pass-throughs:

```python
            va_objective = str(getattr(self.config, "visual_aux_objective", "none") or "none")
            return self.chunked_ce_forward(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                labels=labels,
                use_cache=use_cache,
                chunk_size=loss_chunk_size,
                images=images if (visual_aux_on and va_objective == "aim_pixel") else None,
                image_features=image_features
                if (visual_aux_on and va_objective == "nepa")
                else None,
                image_block_ids=image_block_ids,
            )
```

- [ ] **Step 3: Update the `generate` call site** (lines 457-467) to unpack 7 elements:

```python
            (_, position_ids, attention_mask, _, inputs_embeds, _, _) = (
                self.prepare_inputs_labels_for_multimodal(
                    inputs,
                    position_ids,
                    attention_mask,
                    None,
                    None,
                    image_features,
                    audio_features,
                )
            )
```

- [ ] **Step 4: CPU sanity — existing unit suites still pass**

```bash
python -m pytest tests/test_visual_aux_pairs.py tests/test_model_output.py -q
```
Expected: pass (if `test_model_output.py` needs GPU/weights and skips or errors for environment reasons unrelated to this diff, note it and rely on Task 11).

- [ ] **Step 5: Commit**

```bash
git add src/vlm/models/modeling_vlm.py
git commit -m "feat: image-block-id tracking through the splice + forward threading"
```

---

### Task 5: Visual-aux loss in `chunked_ce_forward`

**Files:**
- Modify: `src/vlm/models/modeling_vlm.py` (`chunked_ce_forward`, lines 265-430)

- [ ] **Step 1: Extend the signature** (lines 265-275):

```python
    def chunked_ce_forward(
        self: Any,
        input_ids: Tensor | None,
        inputs_embeds: Tensor | None,
        attention_mask: Tensor | None,
        position_ids: LongTensor | None,
        past_key_values: list[FloatTensor] | None,
        labels: LongTensor,
        use_cache: bool | None,
        chunk_size: int,
        images: list[Tensor] | None = None,
        image_features: list[Tensor] | None = None,
        image_block_ids: LongTensor | None = None,
    ) -> CausalLMOutputWithPast:
```

- [ ] **Step 2: Read the visual-aux config + generalize hook capture**

Directly after the `aux_active = ...` line (line 305), add:

```python
        # Visual-aux loss (spec 2026-06-06): forward only passes
        # image_block_ids when the head exists, λ > 0 and we're training,
        # so its presence is the single activation signal here.
        va_objective = str(getattr(self.config, "visual_aux_objective", "none") or "none")
        va_weight = float(getattr(self.config, "visual_aux_weight", 0.0) or 0.0)
        va_layer = getattr(self.config, "visual_aux_layer", None)
        va_active = image_block_ids is not None and va_objective != "none" and va_weight > 0.0
```

Change the hook-registration block: replace the loop `for k in aux_layers:` (lines 324-327) so BOTH consumers share the capture machinery —

```python
        capture_layers = sorted(
            set(aux_layers if aux_active else [])
            | ({int(va_layer)} if (va_active and va_layer) else set())
        )
        if capture_layers:
            num_layers = len(self.model.layers)
            # (keep the existing aux_layers range-check block above as-is;
            #  visual_aux_layer is range-checked by train.py)
            for k in capture_layers:
                handles.append(
                    self.model.layers[k - 1].register_forward_hook(_make_capture(k))
                )
```

Note: the existing `if aux_active:` block currently wraps the range-check AND `_make_capture` definition AND the registration loop. Restructure minimally: keep the range-check inside `if aux_active:`; move `_make_capture` out so it is defined whenever `capture_layers` is non-empty; replace the registration loop with the block above.

- [ ] **Step 3: Restructure component stashing + add the visual loss**

Replace the two `self._last_ce_components = {...}` assignments and append the visual block. The full region from `if n_valid == 0:` to `return CausalLMOutputWithPast(...)` becomes:

```python
        components: dict[str, Tensor] = {}
        zero = torch.zeros((), dtype=torch.float32, device=flat_hidden.device)
        if n_valid == 0:
            # Degenerate batch with every target ignored (the reference path
            # would produce NaN): emit an exact-zero loss that still touches
            # lm_head so all trainable params keep grads under deepspeed.
            loss = self.lm_head(flat_hidden[:1]).float().sum() * 0.0
            if aux_active:
                # Keep the log-time component stash rank-symmetric even on
                # all-ignored batches (VLMTrainer.log all-reduces it).
                components["ce_final"] = zero
                components["ce_aux"] = zero
        else:
            hidden_valid = flat_hidden[valid]
            targets_valid = flat_targets[valid]
            total = torch.zeros((), dtype=torch.float32, device=flat_hidden.device)
            for hidden_chunk, target_chunk in zip(
                hidden_valid.split(chunk_size), targets_valid.split(chunk_size), strict=True
            ):
                chunk_logits = self.lm_head(hidden_chunk).float()
                total = total + nn.functional.cross_entropy(
                    chunk_logits, target_chunk, reduction="sum"
                )
            loss = total / n_valid
            if aux_active:
                detach = bool(getattr(self.config, "aux_exit_detach", False))
                final_norm = self.model.norm
                if detach and not hasattr(final_norm, "variance_epsilon"):
                    raise ValueError(
                        "aux_exit_detach=True needs an RMSNorm-style final norm "
                        f"with .variance_epsilon (got {type(final_norm).__name__})"
                    )
                head_weight = (
                    self.lm_head.weight.detach() if detach else self.lm_head.weight
                )
                aux_sum = torch.zeros((), dtype=torch.float32, device=flat_hidden.device)
                for k in aux_layers:
                    h_k = captured.get(k)
                    if h_k is None:
                        raise RuntimeError(
                            f"aux exit layer {k}: forward hook captured nothing"
                        )
                    if torch.is_grad_enabled() and not h_k.requires_grad:
                        raise RuntimeError(
                            f"aux exit layer {k}: captured hidden states are not "
                            "graph-connected (reentrant gradient checkpointing?) — "
                            "the aux loss would silently train nothing"
                        )
                    h_k_valid = h_k.reshape(-1, h_k.shape[-1])[valid]
                    # Same recipe as LayerSkip's forward_early: shared final
                    # norm, then the shared lm_head (h_final is already normed
                    # inside the backbone; raw layer outputs are not).
                    if detach:
                        normed = _rms_norm(
                            h_k_valid, final_norm.weight.detach(), final_norm.variance_epsilon
                        )
                    else:
                        normed = final_norm(h_k_valid)
                    total_k = torch.zeros(
                        (), dtype=torch.float32, device=flat_hidden.device
                    )
                    for hidden_chunk, target_chunk in zip(
                        normed.split(chunk_size), targets_valid.split(chunk_size), strict=True
                    ):
                        chunk_logits = nn.functional.linear(hidden_chunk, head_weight).float()
                        total_k = total_k + nn.functional.cross_entropy(
                            chunk_logits, target_chunk, reduction="sum"
                        )
                    aux_sum = aux_sum + total_k / n_valid
                # ce_aux = unweighted sum of per-exit mean CEs (λ-independent);
                # VLMTrainer.log all-reduces and emits both components.
                components["ce_final"] = loss.detach()
                components["ce_aux"] = aux_sum.detach()
                loss = loss + aux_weight * aux_sum

        if va_active:
            # Visual-aux loss (spec 2026-06-06 §2): predict the NEXT patch of
            # each image block from the hidden state at the current patch.
            # aim_pixel: z-scored pixel MSE (mean over patch dims — sum would
            # silently re-weight CE:visual whenever patch geometry changes).
            # nepa: bidirectional L2-norm negative cosine vs the DETACHED
            # connector embedding (SimSiam stop-grad collapse guard).
            h_for_va = hidden_states
            if va_layer:
                h_k = captured.get(int(va_layer))
                if h_k is None:
                    raise RuntimeError(
                        f"visual_aux_layer {va_layer}: forward hook captured nothing"
                    )
                if torch.is_grad_enabled() and not h_k.requires_grad:
                    raise RuntimeError(
                        f"visual_aux_layer {va_layer}: captured hidden states are "
                        "not graph-connected — the visual aux loss would silently "
                        "train nothing"
                    )
                # Raw layer outputs are unnormed; decode through the shared
                # final norm exactly like the aux-exit branch above.
                h_for_va = self.model.norm(h_k)
            targets_src = images if va_objective == "aim_pixel" else image_features
            num_rows = [int(t.shape[0]) for t in (targets_src or [])]
            flat_pos, segments = build_visual_aux_pairs(
                image_block_ids.to(flat_hidden.device), num_rows
            )
            if not segments:
                # No prediction pairs in this microbatch (text-only / 1-patch
                # images): exact-zero anchor keeps the head's params in the
                # graph every step (deepspeed pattern, same as the n_valid==0
                # lm_head anchor above).
                loss = loss + self.visual_aux_head(flat_hidden[:1]).float().sum() * 0.0
                components["visual_aux"] = zero
                if va_objective == "nepa":
                    components["visual_aux_cos"] = zero
                    components["visual_aux_tgt_std"] = zero
            else:
                flat_va_hidden = h_for_va.reshape(-1, h_for_va.shape[-1])
                preds_in = flat_va_hidden[flat_pos]
                targets = prepare_visual_aux_targets(va_objective, targets_src, segments).to(
                    preds_in.device
                )
                n_pairs = preds_in.shape[0]
                va_total = torch.zeros((), dtype=torch.float32, device=flat_hidden.device)
                cos_total = torch.zeros((), dtype=torch.float32, device=flat_hidden.device)
                for pred_chunk, target_chunk in zip(
                    preds_in.split(chunk_size), targets.split(chunk_size), strict=True
                ):
                    pred = self.visual_aux_head(pred_chunk).float()
                    if va_objective == "aim_pixel":
                        va_total = va_total + (pred - target_chunk).pow(2).mean(dim=-1).sum()
                    else:
                        pred = nn.functional.normalize(pred, dim=-1)
                        cos = (pred * target_chunk).sum(dim=-1)
                        cos_total = cos_total + cos.sum()
                        va_total = va_total - cos.sum()
                va_loss = va_total / n_pairs
                loss = loss + va_weight * va_loss
                # Unweighted (λ-independent) component + nepa collapse alarms:
                # cos → 1 with tgt_std → 0 is the collapse signature.
                components["visual_aux"] = va_loss.detach()
                if va_objective == "nepa":
                    components["visual_aux_cos"] = (cos_total / n_pairs).detach()
                    components["visual_aux_tgt_std"] = targets.std(dim=0).mean().detach()

        if components:
            self._last_ce_components = components
        return CausalLMOutputWithPast(
            loss=loss,
            logits=None,
            past_key_values=outputs.past_key_values,
        )
```

(The CE/aux-exit math is copied verbatim from the current implementation — only the stash targets changed from direct `self._last_ce_components = {...}` to the `components` dict. Diff carefully against the current file to confirm nothing else moved.)

- [ ] **Step 4: CPU sanity**

```bash
python -m pytest tests/test_visual_aux_pairs.py -q && python devtools/lint.py 2>/dev/null || ruff check src/vlm/models/modeling_vlm.py
```
Expected: tests pass; no new lint errors in the touched file.

- [ ] **Step 5: Commit**

```bash
git add src/vlm/models/modeling_vlm.py
git commit -m "feat: visual-aux loss in chunked_ce_forward (aim_pixel MSE / nepa cosine)"
```

---

### Task 6: Generalize component logging in `VLMTrainer.log`

**Files:**
- Modify: `src/vlm/train/vlm_trainer.py` (lines 114-128)

- [ ] **Step 1: Replace the hard-coded two-key block**

Replace lines 119-128 (`comps = getattr(...)` through `logs["ce_aux"] = float(vals[1])`) with:

```python
            comps = getattr(self.model, "_last_ce_components", None)
            if comps is not None:
                # Key set is config-determined (aux-exit and/or visual-aux),
                # identical on every rank and stashed every step (zeros on
                # degenerate batches) — sorted iteration keeps the collective
                # rank-deterministic.
                keys = sorted(comps)
                vals = torch.stack([comps[k] for k in keys]).to(self.args.device)
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(vals)
                    vals = vals / self.args.world_size
                for i, key in enumerate(keys):
                    logs[key] = float(vals[i])
```

(Behavior for aux-exit-only runs is unchanged: same two keys, same values; the comment above the block at lines 114-118 stays.)

- [ ] **Step 2: Commit**

```bash
git add src/vlm/train/vlm_trainer.py
git commit -m "train: generalize loss-component logging to arbitrary stashed keys"
```

---

### Task 7: Optimizer group + trainability for the head

**Files:**
- Modify: `src/vlm/train/set_trainable.py` (component_prefixes line 38-47, set_trainable_params line 104-129)
- Modify: `src/vlm/train/optimizer.py` (component_to_config line 25-30, component_configs line 75-88)

- [ ] **Step 1: Register the prefix** — in `group_params_by_prefix`, add to `component_prefixes` (after the `"lm_head"` entry):

```python
        # Visual-aux head (spec 2026-06-06): its own group — without this the
        # params would fall through to "language_model" (the unassigned-prefix
        # default below) and silently take the LM lr / freeze flag.
        "visual_aux_head": ["visual_aux_head"],
```

- [ ] **Step 2: Always-trainable** — in `set_trainable_params`, after the `train_connector` block (line 121-123), add:

```python
    # The visual-aux head exists only when the objective is active and is
    # always trainable — it is the fresh module the aux loss exists to train,
    # in every recipe (incl. frozen-trunk retrofits later).
    for _, param in grouped_params.get("visual_aux_head", []):
        param.requires_grad = True
```

- [ ] **Step 3: Optimizer wiring** — in `optimizer.py`:

`configure_optimizers`'s `component_to_config` (line 25-30) gains:

```python
        "visual_aux_head": "visual_aux_head",
```

`build_optimizer_params`'s `component_configs` (line 75-88) gains:

```python
        "visual_aux_head": {
            # None falls back to the run's default lr / the LM weight decay —
            # the head is fresh, so a higher dedicated lr is a legitimate dial.
            "weight_decay": trainer_config.visual_aux_head_wd
            if trainer_config.visual_aux_head_wd is not None
            else trainer_config.language_model_wd,
            "learning_rate": trainer_config.visual_aux_head_lr
            if trainer_config.visual_aux_head_lr is not None
            else trainer_config.learning_rate,
        },
```

(`trainer_config.learning_rate` is HF `TrainingArguments.learning_rate`, set from `default_lr` in `get_training_args` — verified at `training_arguments.py:281`.)

- [ ] **Step 4: Commit**

```bash
git add src/vlm/train/set_trainable.py src/vlm/train/optimizer.py
git commit -m "train: visual_aux_head optimizer group (own lr/wd, always trainable)"
```

---

### Task 8: train.py validation + config copy

**Files:**
- Modify: `src/vlm/train/train.py` (new validator next to `validate_aux_exit_config` line 31-54; copy block in `train()` after line 129)

- [ ] **Step 1: Add the validator** (below `validate_aux_exit_config`):

```python
def validate_visual_aux_config(
    objective: Any,
    layer: Any,
    num_hidden_layers: int,
    loss_chunk_size: int,
    encoder_free: bool,
) -> tuple[str, int | None]:
    """Validate the visual-aux dials (spec:
    docs/superpowers/specs/2026-06-06-visual-aux-loss-design.md). Returns the
    normalized (objective, layer) as plain python types (config.json-safe)."""
    objective = str(objective or "none")
    if objective == "none":
        return "none", None
    if objective not in ("aim_pixel", "nepa"):
        raise ValueError(
            f"model.visual_aux.objective {objective!r} not in (none, aim_pixel, nepa)"
        )
    if loss_chunk_size <= 0:
        raise ValueError(
            "model.visual_aux.objective requires trainer.loss_chunk_size > 0 — the "
            "visual aux loss is implemented only in the chunked-CE training path"
        )
    if not encoder_free:
        raise ValueError(
            "visual_aux supports only the encoder-free (raw_patch) path: targets "
            "are raw patch rows / connector embedding rows, which the classic "
            "vision-tower path does not produce"
        )
    if layer is None:
        return objective, None
    k = int(layer)
    if not 1 <= k <= num_hidden_layers - 1:
        raise ValueError(
            f"trainer.visual_aux_layer {k} out of range [1, {num_hidden_layers - 1}] "
            f"for a {num_hidden_layers}-layer backbone"
        )
    return objective, k
```

- [ ] **Step 2: Wire into `train()`** — after the aux-exit warning block (line 121-129), add:

```python
    # Visual-aux (spec 2026-06-06): objective/head dials were placed on
    # model.config at build time (load_model); validate them against the real
    # backbone and copy the trainer-side dials next to them.
    va_objective, va_layer = validate_visual_aux_config(
        getattr(model.config, "visual_aux_objective", "none"),
        training_args.visual_aux_layer,
        num_hidden_layers=len(model.model.layers),
        loss_chunk_size=int(training_args.loss_chunk_size),
        encoder_free=model.model.vision_model is None,
    )
    model.config.visual_aux_weight = float(training_args.visual_aux_weight)
    model.config.visual_aux_layer = va_layer
    if va_objective != "none" and getattr(model, "visual_aux_head", None) is None:
        raise ValueError(
            "visual_aux objective is set but the model has no visual_aux_head — "
            "was the model loaded from an understanding-only checkpoint?"
        )
    if va_objective != "none" and model.config.visual_aux_weight <= 0.0:
        # Loud, mirroring the aux-exit guard: this arm would silently train
        # as a pure baseline duplicate — almost certainly a run-config typo.
        log.warning(
            "visual_aux_objective=%s but visual_aux_weight=%s: the visual aux loss "
            "is DISABLED (weight must be > 0) — this run is an exact baseline duplicate",
            va_objective,
            model.config.visual_aux_weight,
        )
```

- [ ] **Step 3: Commit**

```bash
git add src/vlm/train/train.py
git commit -m "train: visual-aux validation + model.config copy"
```

---

### Task 9: Mirror the head in the export templates

**Files:**
- Modify: `templates/modeling_vlm.py.j2` (causal `__init__` ~line 38 region; add `_build_visual_aux_head`)

- [ ] **Step 1: Mirror the structural change only.** Exported checkpoints need the head MODULE so `from_pretrained` loads its weights; the training-only loss/splice changes are NOT needed in the export (the template's chunked path has no aux-exit either — same precedent). In `templates/modeling_vlm.py.j2`: locate the causal-class `__init__` (search for `self.lm_head = nn.Linear`), add `self.visual_aux_head = self._build_visual_aux_head(config)` before `self.post_init()`, and add the `_build_visual_aux_head` method (copy the exact body from Task 3 Step 1, adjusted to the template's method style — it defines plain methods on the class, not factory closures; check how `_build_audio_connector` appears at template line 90 and match that form). `templates/configuration_vlm.py.j2` needs NO change — the `visual_aux_*` keys ride the existing kwargs passthrough (the `conversation_version` precedent).

- [ ] **Step 2: Render check** — confirm the template still renders/compiles. Look at how the push path renders it:

```bash
grep -rn "modeling_vlm.py.j2" src/ --include=*.py | head -3
```

then exercise the render function it points to with a python one-liner (e.g. jinja2 `Template(open('templates/modeling_vlm.py.j2').read()).render(...)` with the variables the push code passes — copy them from the push code) and `compile()` the result. Expected: renders and compiles without `SyntaxError`.

- [ ] **Step 3: Commit**

```bash
git add templates/modeling_vlm.py.j2
git commit -m "templates: visual_aux_head in exported modeling code"
```

---

### Task 10: Arm configs + spec refinement note

**Files:**
- Create: `src/vlm/config/sft-unified-aimpixel.yaml`
- Create: `src/vlm/config/sft-unified-nepa.yaml`
- Modify: `docs/superpowers/specs/2026-06-06-visual-aux-loss-design.md` (§3.5 table)

- [ ] **Step 1: Read the sibling override config** `src/vlm/config/sft-unified-earlyfusion.yaml` and mirror its exact defaults-list idiom (it is the proven aux-exit arm pattern; `test_aux_exit.py` test 0 composes it).

- [ ] **Step 2: Create `sft-unified-aimpixel.yaml`:**

```yaml
defaults:
    - sft-unified
    - _self_

# Visual-aux arm 1 (spec 2026-06-06): next-patch PIXEL prediction at image
# positions (AIM/AIMv2-style; per-patch z-scored MSE through a fresh 2-layer
# MLP head). Same data order/seed as the sft-unified baseline.
# Launch: CONFIG_NAME=sft-unified-aimpixel sbatch train.slurm
model:
    visual_aux:
        objective: aim_pixel

trainer:
    visual_aux_weight: 0.5
```

- [ ] **Step 3: Create `sft-unified-nepa.yaml`:**

```yaml
defaults:
    - sft-unified
    - _self_

# Visual-aux arm 2 (spec 2026-06-06): next-patch EMBEDDING prediction (NEPA,
# arXiv:2512.16922; stop-grad connector-output target, L2-norm cosine).
# Collapse alarms land in wandb as visual_aux_cos / visual_aux_tgt_std —
# cos -> 1 with tgt_std -> 0 is the collapse signature.
# Launch: CONFIG_NAME=sft-unified-nepa sbatch train.slurm
model:
    visual_aux:
        objective: nepa

trainer:
    visual_aux_weight: 0.5
```

(If `sft-unified-earlyfusion.yaml` shows extra required idiom — e.g. an `hydra:` block or `trainer.name` override — replicate it identically in both files.)

- [ ] **Step 4: Compose check:**

```bash
python -c "
from hydra import compose, initialize_config_dir
import vlm.config as c; c.register_configs()
with initialize_config_dir(config_dir='$PWD/src/vlm/config', version_base=None):
    for name, obj in (('sft-unified-aimpixel','aim_pixel'), ('sft-unified-nepa','nepa')):
        cfg = compose(config_name=name)
        assert cfg.model.visual_aux.objective == obj, (name, cfg.model.visual_aux.objective)
        assert abs(cfg.trainer.visual_aux_weight - 0.5) < 1e-9
        assert cfg.trainer.loss_chunk_size == 1024  # inherited from sft-unified
        print(name, 'OK')
"
```
Expected: both `OK` lines.

- [ ] **Step 5: Record the design refinement in the spec** — in §3.5's table, replace the `config_schema.py` row's field list with the split actually implemented (`model.visual_aux.{objective,head_depth,head_hidden}` structural + `trainer.{visual_aux_weight,visual_aux_layer,visual_aux_head_lr,visual_aux_head_wd}`), and change the `configuration_vlm.py + .j2` row to "no change needed — `visual_aux_*` keys ride the config kwargs passthrough (`conversation_version` precedent)". Add one line at the bottom of §3.5: "Refinement during implementation (2026-06-06): structural dials moved from TrainerConfig to `model.visual_aux` so the causal `__init__` can build the head from the config it is constructed with."

- [ ] **Step 6: Commit**

```bash
git add src/vlm/config/sft-unified-aimpixel.yaml src/vlm/config/sft-unified-nepa.yaml docs/superpowers/specs/2026-06-06-visual-aux-loss-design.md
git commit -m "config: visual-aux ablation arm yamls (aim_pixel / nepa, lambda 0.5)"
```

---

### Task 11: GPU integration test (the pre-launch gate)

**Files:**
- Create: `devtools/test_visual_aux.py` (mirror `devtools/test_aux_exit.py`'s harness — real 0.6B model via `load_model`, fp32+sdpa, batch from `test_chunked_ce.build_batch`)

- [ ] **Step 1: Write the test.** Structure (follow `test_aux_exit.py` shape; reuse its imports, `rel_diff`, srun header):

```python
"""Tests for the visual-aux losses (aim_pixel / nepa next-patch prediction).
Spec: docs/superpowers/specs/2026-06-06-visual-aux-loss-design.md.

Run on a GPU node (from the repo or worktree root):
    srun -p ckpt-all -A cse-ckpt --gpus=l40:1 --mem=48G --time=0:30:00 \
        bash -c 'source /mmfs1/gscratch/krishna/leoym/small-vlm/.venv/bin/activate \
                 && HF_HUB_OFFLINE=1 python devtools/test_visual_aux.py'

Checks (fp32+sdpa, real multimodal batch from test_chunked_ce.build_batch):
  0. Both arm configs compose (objective, lambda 0.5, chunk inherited).
  1. Baseline parity: objective=none -> no head attribute, chunked loss ==
     full-logits reference, grads match (protects the live baseline).
  2. aim_pixel numerical correctness: loss == L_CE_ref + 0.5 * naive
     reference (full-precision pairs/z-score/MSE on captured hiddens),
     components logged.
  3. nepa numerical correctness: same construction with cosine; alarms
     (visual_aux_cos, visual_aux_tgt_std) logged and consistent.
  4. Gradient routing: nepa target stop-grad — with ONLY the connector
     trainable and lambda > 0, connector grads under nepa equal
     baseline-CE connector grads PLUS a prediction-path contribution; the
     direct check is that targets are detached (unit-tested in
     tests/test_visual_aux_pairs.py) — here assert head grads are nonzero
     and lm_head grads are bit-identical to the objective=none run.
  5. Degenerate batch (text-only): finite loss, head grads exist (zeros),
     zero components stashed.
  6. visual_aux_layer=6 + gradient checkpointing: same loss as without
     checkpointing; backward succeeds.
  7. Optimizer grouping: visual_aux_head params land in their own group
     with the configured lr; set_trainable keeps them trainable with
     train_language_model=False.
  8. validate_visual_aux_config rejects bad objective / chunk=0 /
     encoder-present / layer out of range.
"""
```

Key implementation notes for the test body (write real code, modeled line-by-line on `test_aux_exit.py`):
- Build the model ONCE per objective by composing `sft-unified` with overrides `["model=qwen3-0.6b-unified", "trainer.bf16=false", "trainer.attn_implementation=sdpa", "model.visual_aux.objective=aim_pixel"]` (and `nepa`), via `load_model` — the head is then built by `__init__`. For the baseline use objective `none`.
- After `load_model`, set the runtime knobs directly (the test bypasses `train()`): `model.config.loss_chunk_size = 1024`, `model.config.visual_aux_weight = 0.5`, `model.config.visual_aux_layer = None`.
- Reference computation for tests 2-3: monkeypatch `model.prepare_inputs_labels_for_multimodal` to capture the 7-tuple (labels + block ids); hook `model.model.norm` to capture the post-norm last hidden state; then compute the naive reference with `build_visual_aux_pairs` + `prepare_visual_aux_targets` + the REAL `model.visual_aux_head` in full precision, and assert `loss_on == loss_ce_ref + 0.5 * va_ref` within 2e-5. Note the batch's `images` list — pass the same list the batch carries.
- Test 5: build a text-only batch (reuse `build_batch`'s text-only sample path, or construct input_ids/labels without `<image>` and a dummy image entry like the collator does).
- Test 7: call `group_params_by_prefix(model)` and `configure_optimizers(model, args)` with a `TrainingArguments`-like namespace carrying `visual_aux_head_lr=1e-4`; assert the head's params are in a group with `lr == 1e-4` and that after `set_trainable_params(model, ns(train_language_model=False, train_connector=False, train_vision_model=False))` the head params still have `requires_grad=True`.

- [ ] **Step 2: Run on a GPU node**

```bash
srun -p ckpt-all -A cse-ckpt --gpus=l40:1 --mem=48G --time=0:30:00 \
  bash -c 'source /mmfs1/gscratch/krishna/leoym/small-vlm/.venv/bin/activate \
           && HF_HUB_OFFLINE=1 python devtools/test_visual_aux.py'
```
Expected: `ALL VISUAL-AUX TESTS PASSED`. Iterate until green — **this is the gate before any baseline requeue window** (the running jobs relaunch from this tree).

- [ ] **Step 3: Re-run the aux-exit suite (regression on the shared code paths)**

```bash
srun -p ckpt-all -A cse-ckpt --gpus=l40:1 --mem=48G --time=0:30:00 \
  bash -c 'source /mmfs1/gscratch/krishna/leoym/small-vlm/.venv/bin/activate \
           && HF_HUB_OFFLINE=1 python devtools/test_aux_exit.py'
```
Expected: `ALL AUX-EXIT TESTS PASSED` (the stash/hook restructure must not move aux-exit numerics).

- [ ] **Step 4: Commit**

```bash
git add devtools/test_visual_aux.py
git commit -m "test: visual-aux GPU integration suite (parity, numerics, routing)"
```

---

### Task 12: Lint + wrap-up

- [ ] **Step 1: Repo lint/format over touched files**

```bash
ruff check src/vlm tests devtools/test_visual_aux.py && ruff format --check src/vlm tests devtools/test_visual_aux.py
```
(Or `python devtools/lint.py` if that is the repo's lint entry — check its header.) Fix anything it flags in the touched files only; commit as `chore: lint`.

- [ ] **Step 2: Full CPU test sweep**

```bash
python -m pytest tests/ -q
```
Expected: no new failures vs the pre-change baseline (run `git stash && python -m pytest tests/ -q && git stash pop` first if a baseline comparison is needed).

- [ ] **Step 3: Final review checklist**
  - `git log --oneline` shows one commit per task, no unrelated files swept in.
  - `grep -n "visual_aux" src/vlm/config/sft-unified.yaml` → no hits (baseline yaml untouched).
  - Launch instructions confirmed in both arm yamls' headers (`CONFIG_NAME=... sbatch train.slurm`).
