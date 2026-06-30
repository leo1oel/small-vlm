"""Behavior-preservation + truthful-labeling guards for the S1/S2 unfreeze
recipes and the visual-expert logging group (train/set_trainable.py).

The three unfreeze flags (train_vision_model / train_language_model /
train_connector) govern ONLY the base transformer trunk. The per-layer visual
experts (FFN mlp_visual / norm norm_visual / attention proj_visual + their
sigmoid gates) are substring-named inside the decoder layers, so they cannot be
prefix-grouped; they are force-trained in set_trainable_params regardless of
train_language_model and are now counted under their own `visual_expert`
logging group instead of hiding inside `language_model`.

These tests prove the regrouping is byte-identical in trainability (no behavior
change) for BOTH the pretrain (S1) and finetune (S2) recipes, and guard the
truthful label (visual_expert trainable, language_model frozen under S1).
"""

import types
from collections import defaultdict
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from transformers import AutoConfig

import vlm
import vlm.train.optimizer as opt_mod
import vlm.train.set_trainable as st
from vlm.models import get_dynamic_vlm
from vlm.models.modeling_vlm import init_visual_experts_from_text, is_visual_expert_param
from vlm.train.optimizer import configure_optimizers

UNFREEZE_DIR = Path(vlm.__file__).parent / "config" / "trainer" / "unfreeze"

BASE_LM = "Qwen/Qwen3-0.6B"
_VISION_DIALS = {"patch_size": 4, "pooling_kernel_size": 1, "max_soft_tokens": 16}
_PATCH_DIM = (4 * 1) ** 2 * 3  # 48


def _build_tiny_vlm():
    """A tiny CPU encoder-free VLM with the FFN + norm visual experts wired from
    config -- enough structure (per-layer visual experts, connector, text trunk)
    to exercise set_trainable_params end to end. Mirrors the builder in
    tests/test_visual_experts.py but is kept self-contained here so this module
    needs no sibling-test import."""
    base_cfg = AutoConfig.from_pretrained(BASE_LM)
    base_cfg.hidden_size = 32
    base_cfg.intermediate_size = 64
    base_cfg.num_hidden_layers = 2
    base_cfg.layer_types = ["full_attention"] * 2
    base_cfg.num_attention_heads = 4
    base_cfg.num_key_value_heads = 2
    base_cfg.head_dim = 8
    base_cfg.max_position_embeddings = 512

    VLMForCausalLM, VLMConfig = get_dynamic_vlm(BASE_LM)
    config = VLMConfig(
        hf_name=BASE_LM,
        vision_config={**_VISION_DIALS, "hf_name": None, "hidden_size": _PATCH_DIM},
        connector_config={
            "name": "raw_patch",
            "type": "raw_patch",
            "mm_embed_dim": 32,
            "mm_posemb_size": _VISION_DIALS["max_soft_tokens"],
        },
        audio_config=None,
        **base_cfg.to_dict(),
        image_token="<image>",
        image_token_index=-200,
        audio_token="<audio>",
        audio_token_index=-201,
        ignore_index=-100,
        max_seq_length=512,
        padding_side="left",
        use_start_end_tokens=False,
        image_start_token="<im_start>",
        image_end_token="<im_end>",
        conversation_version="qwen_2_5",
        # the FFN + norm visual experts (init-from-text, ungated)
        visual_expert=True,
        visual_expert_ffn=True,
        visual_expert_norm=True,
        visual_expert_attention=False,
        visual_expert_gate=False,
        visual_expert_layers=None,
        visual_expert_init_from_text=True,
    )
    torch.manual_seed(0)
    model = VLMForCausalLM(config)
    init_visual_experts_from_text(model.model)
    model.eval()
    return model


def _load_unfreeze(name: str):
    """Load the real shipped unfreeze recipe (ties these tests to the actual
    config files, so a flag flip there is caught here)."""
    return OmegaConf.load(UNFREEZE_DIR / f"{name}.yaml")


def _old_group_params_by_prefix(model):
    """Faithful copy of group_params_by_prefix as it was BEFORE the
    `visual_expert` special-case was added -- i.e. visual experts fall through
    to "language_model". Used to snapshot the pre-change trainability so the
    regrouping can be proven behavior-identical. set_trainable_params itself is
    unchanged; only the grouping it consults differs."""
    component_prefixes = {
        "vision_model": ["model.vision_model", "model.vision_tower", "vision_tower"],
        "language_model": [],
        "connector": ["model.connector", "model.audio_connector", "connector"],
        "lm_head": ["lm_head"],
        "visual_aux_head": ["visual_aux_head"],
        "generation": ["gen_x_head", "gen_t_embed", "gen_patch_embed"],
        "learnable_query": ["learnable_query"],
        "visual_distill_head": ["visual_distill_head"],
    }
    grouped_params = defaultdict(list)
    for name, param in model.named_parameters():
        assigned = False
        for component, prefixes in component_prefixes.items():
            if any(name.startswith(prefix) for prefix in prefixes):
                grouped_params[component].append((name, param))
                assigned = True
                break
        if not assigned and not any(
            name.startswith(prefix)
            for prefixes in [component_prefixes["vision_model"], component_prefixes["connector"]]
            for prefix in prefixes
        ):
            grouped_params["language_model"].append((name, param))
    return grouped_params


def _snapshot(model):
    return {n: p.requires_grad for n, p in model.named_parameters()}


@pytest.mark.parametrize("recipe", ["pretrain", "finetune"])
def test_trainable_set_unchanged_by_regrouping(recipe, monkeypatch):
    """The exact {name: requires_grad} set must be identical with the new
    `visual_expert` grouping and the old grouping, for BOTH recipes -- proving
    the regrouping is a pure relabeling with zero trainability change."""
    cfg = _load_unfreeze(recipe)
    model = _build_tiny_vlm()

    # NEW grouping (the code under test): visual experts get their own group.
    st.set_trainable_params(model, cfg)
    new_snapshot = _snapshot(model)
    assert any(is_visual_expert_param(n) for n in new_snapshot), "test model has no visual experts"

    # OLD grouping: visual experts fall through to "language_model".
    # set_trainable_params resets all requires_grad each call, so this fully
    # recomputes the trainable set from scratch with the pre-change grouping.
    monkeypatch.setattr(st, "group_params_by_prefix", _old_group_params_by_prefix)
    st.set_trainable_params(model, cfg)
    old_snapshot = _snapshot(model)

    assert new_snapshot.keys() == old_snapshot.keys()
    diffs = {k for k in new_snapshot if new_snapshot[k] != old_snapshot[k]}
    assert not diffs, f"regrouping changed trainability under {recipe} recipe: {diffs}"

    # And the visual experts ARE trained under both recipes (force-on).
    assert all(new_snapshot[n] for n in new_snapshot if is_visual_expert_param(n)), (
        "visual experts must be force-trained in every recipe"
    )


def _trainer_config():
    """A trainer config with a DISTINCT lr/wd per bucket, so each param's
    optimizer assignment is identifiable (and the LM bucket is distinguishable
    from the connector bucket)."""
    return types.SimpleNamespace(
        visual_encoder_lr=1.0,
        visual_encoder_wd=0.01,
        language_model_lr=2.0,
        language_model_wd=0.02,
        connector_lr=3.0,
        connector_wd=0.03,
        visual_aux_head_lr=None,
        visual_aux_head_wd=None,
        learning_rate=9.0,
    )


def _opt_assignment(model, trainer_config):
    """{param_name: (lr, weight_decay)} for every param the optimizer will step.
    A param missing from this map is one configure_optimizers silently dropped
    (trainable but never stepped)."""
    id_to_name = {id(p): n for n, p in model.named_parameters()}
    out = {}
    for group in configure_optimizers(model, trainer_config):
        for param in group["params"]:
            out[id_to_name[id(param)]] = (group["lr"], group["weight_decay"])
    return out


@pytest.mark.parametrize("recipe", ["pretrain", "finetune"])
def test_optimizer_assignment_unchanged_by_regrouping(recipe, monkeypatch):
    """configure_optimizers SILENTLY DROPS any param group it doesn't map, so
    the regrouping must (a) keep every visual-expert param in the optimizer at
    the LM lr/wd and (b) produce a byte-identical {param: (lr, wd)} assignment
    vs the old grouping -- guarding the trainable-but-never-stepped trap."""
    tc = _trainer_config()
    model = _build_tiny_vlm()

    # NEW grouping + the new visual_expert -> "model" mapping.
    st.set_trainable_params(model, _load_unfreeze(recipe))
    new = _opt_assignment(model, tc)

    expert_assign = {n: lrwd for n, lrwd in new.items() if is_visual_expert_param(n)}
    assert expert_assign, "visual experts missing from the optimizer (silently dropped!)"
    assert all(lr == tc.language_model_lr for (lr, _wd) in expert_assign.values()), (
        "visual experts must train at the LM lr (the 'model' bucket)"
    )

    # OLD grouping (experts fall into "language_model", also -> "model"). Patch
    # the name in BOTH modules: set_trainable_params and configure_optimizers
    # each hold their own binding.
    monkeypatch.setattr(st, "group_params_by_prefix", _old_group_params_by_prefix)
    monkeypatch.setattr(opt_mod, "group_params_by_prefix", _old_group_params_by_prefix)
    st.set_trainable_params(model, _load_unfreeze(recipe))
    old = _opt_assignment(model, tc)

    diffs = {k: (new.get(k), old.get(k)) for k in set(new) | set(old) if new.get(k) != old.get(k)}
    assert not diffs, f"optimizer assignment changed under {recipe} recipe: {diffs}"


def test_pretrain_breakdown_labels_visual_expert_not_language_model():
    """Misleading-label regression guard: under the S1 (connector-only) recipe
    the force-trained visual experts must be reported under their own
    `visual_expert` group as trainable, and `language_model` (the text trunk)
    must read frozen -- NOT "language_model: ...B trainable"."""
    cfg = _load_unfreeze("pretrain")
    assert cfg.train_language_model is False, "S1 recipe must freeze the text trunk"
    model = _build_tiny_vlm()

    stats = st.set_trainable_params(model, cfg)
    comps = stats["components"]

    # visual_expert: present and fully trained (force-on under a frozen LM).
    assert "visual_expert" in comps, "visual experts must have their own logging group"
    ve_trainable, ve_total = comps["visual_expert"][""]
    assert ve_total > 0, "test model should have visual-expert params"
    assert ve_trainable == ve_total, "all visual-expert params are force-trained in S1"

    # language_model: the text trunk is present but frozen (the truthful label).
    lm_trainable, lm_total = comps["language_model"][""]
    assert lm_total > 0, "text trunk should still be a non-empty group"
    assert lm_trainable == 0, "text trunk must read frozen under the S1 recipe"
