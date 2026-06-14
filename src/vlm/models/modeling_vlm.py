import json
import logging
from pathlib import Path
from typing import Any, override

import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor, Tensor
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_MAPPING,
)

from . import xmodal_mask as _xmodal_mask  # noqa: F401  (registers sdpa_xmodal)
from .configuration_vlm import create_dynamic_vlm_config_class
from .connectors import Connector, connector_map

log: logging.Logger = logging.getLogger(name=__name__)


def _rms_norm(hidden: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Functional replica of Qwen3RMSNorm.forward (transformers 5.10.2,
    models/qwen3/modeling_qwen3.py): fp32 upcast -> x * rsqrt(mean-square +
    eps) -> downcast to the input dtype -> THEN scale by weight (the order
    matters for bit-parity). Exists so the aux-exit detach path can decode
    through `weight.detach()` without touching the shared module; pinned to
    the real module by devtools/test_aux_exit.py."""
    input_dtype = hidden.dtype
    h = hidden.to(torch.float32)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return weight * h.to(input_dtype)


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

    Precondition: `segments` is non-empty — callers gate on it (the no-pairs
    microbatch takes the zero-loss anchor path in chunked_ce_forward instead).

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


def get_dynamic_vlm_class(
    base_language_model_name_or_path: str,  # e.g., "google/gemma-3-4b-it"
):
    # Determine the base LLM's actual config class and model class
    try:
        base_language_model_actual_config = AutoConfig.from_pretrained(
            base_language_model_name_or_path, trust_remote_code=True
        )
        base_language_model_config_class = base_language_model_actual_config.__class__
    except Exception as _:
        config_file = Path(base_language_model_name_or_path) / "config.json"
        try:
            data = json.loads(config_file.read_text())
            base_language_model_name_or_path = data["hf_name"]
            base_language_model_actual_config = AutoConfig.from_pretrained(
                base_language_model_name_or_path, trust_remote_code=True
            )
            base_language_model_config_class = base_language_model_actual_config.__class__
        except Exception as e:
            raise e

    if base_language_model_config_class not in MODEL_FOR_CAUSAL_LM_MAPPING:
        raise ValueError(
            f"Config class {base_language_model_config_class.__name__} for LLM {base_language_model_name_or_path} "
            f"not in MODEL_FOR_CAUSAL_LM_MAPPING. Cannot determine parent CausalLM class."
        )
    ParentLLMClass = MODEL_MAPPING[base_language_model_config_class]
    ParentCausalLLMClass = MODEL_FOR_CAUSAL_LM_MAPPING[base_language_model_config_class]
    log.info(
        f"Determined ParentLLMClass and ParentCausalLLMClass for VLM: {ParentLLMClass.__name__} and {ParentCausalLLMClass.__name__} (from {base_language_model_name_or_path})"
    )
    return ParentLLMClass, ParentCausalLLMClass, base_language_model_name_or_path


def create_dynamic_vlm_class(
    base_language_model_name_or_path: str,  # e.g., "google/gemma-3-4b-it"
    config_class: Any,
    ParentLLMClass: Any,
):
    @override
    def __init__(self: Any, config):  # pyright: ignore
        super(self.__class__, self).__init__(config)
        # Visual FFN experts (spec 2026-06-14): attach the per-layer
        # modality-routed expert structure on BOTH paths (no-op unless
        # config.visual_expert), so checkpoint shards have a module to load
        # into; the text->visual weight copy is fresh-build-only (vlm.py).
        install_visual_experts(self, config)
        if not config.lazy_load:
            self.vision_model = self._build_vision_model(config)
            self.connector = self._build_connector(config)
            self.audio_connector = self._build_audio_connector(config)
        log.info(f"DynamicVLM class {self.__class__.__name__} initialized.")

    def init_other_components(self: Any) -> None:
        self.vision_model = self._build_vision_model(self.config)
        self.connector = self._build_connector(self.config)
        self.audio_connector = self._build_audio_connector(self.config)

    def _build_vision_model(self: Any, config: Any) -> PreTrainedModel | None:
        vision_config = config.vision_config
        if getattr(vision_config, "hf_name", None) is None:
            # Encoder-free path (gemma4_unified-style, connector type "raw_patch"):
            # there is no vision tower at all — raw patches from RawImageProcessor
            # go straight to the connector. Returning None here covers both the
            # fresh-build path (init_other_components) and the checkpoint-reload
            # path (__init__ under meta device), since both funnel through this
            # builder. Downstream, encode dispatch checks `vision_model is None`.
            return None
        # transformers v5 instantiates the model under a `torch.device("meta")`
        # context during `from_pretrained` (modeling_utils: `with torch.device("meta")`).
        # A nested `from_pretrained` inside that context is explicitly forbidden
        # (integrations/accelerate.check_and_set_device_map raises). This branch is
        # hit when RELOADING a saved VLM checkpoint (lazy_load=False -> vision tower
        # built in __init__): there the vision weights already live in the VLM
        # checkpoint, so we only need the module *structure* and let the outer
        # from_pretrained populate the weights. Build with `from_config` (no weight
        # download) under a meta context; use `from_pretrained` only on the
        # fresh-build path (init_other_components, run outside any meta context).
        under_meta = torch.tensor([]).device.type == "meta"
        if under_meta:
            hf_vision_config = AutoConfig.from_pretrained(
                vision_config.hf_name, trust_remote_code=True
            )
            if getattr(hf_vision_config, "vision_config", None) is not None:
                hf_vision_config = hf_vision_config.vision_config
            hf_vision_config._attn_implementation = "sdpa"  # pyright: ignore
            visual_encoder: PreTrainedModel = AutoModel.from_config(
                hf_vision_config, trust_remote_code=True
            )
        else:
            visual_encoder = AutoModel.from_pretrained(
                vision_config.hf_name,
                trust_remote_code=True,
                # Pin the vision tower to sdpa explicitly (matches the current
                # transformers default for these encoders; guards against future
                # library default changes). Note: a VisionConfig field would NOT
                # work here — PretrainedConfig.__init__ pops 'attn_implementation'
                # into the private _attn_implementation, so a schema knob would be
                # silently dead.
                attn_implementation="sdpa",
            )
        if getattr(visual_encoder, "vision_model", None):
            visual_encoder = visual_encoder.vision_model  # pyright: ignore
        return visual_encoder

    def _build_connector(self: Any, config: Any) -> Connector:
        connector_config = config.connector_config
        connector_class = connector_map.get(connector_config.type)
        if not connector_class:
            raise ValueError(f"Unsupported connector type: {connector_config.type}")
        return connector_class(
            connector_config,
            self.config.vision_config.hidden_size,
            config.hidden_size,
        )

    def _build_audio_connector(self: Any, config: Any) -> Connector | None:
        """Audio pathway (encoder-free, gemma4_unified-style): a connector that
        projects raw waveform frames into LM space. None when the config has no
        audio_config or it is disabled — vision-only models stay audio-free."""
        audio_config = getattr(config, "audio_config", None)
        if audio_config is None or not getattr(audio_config, "enabled", True):
            return None
        connector_class = connector_map.get(audio_config.type)
        if not connector_class:
            raise ValueError(f"Unsupported audio connector type: {audio_config.type}")
        # For raw_waveform the "image_hidden_size" slot carries the audio frame
        # size (samples_per_token); derived from the audio config, never hand-set.
        return connector_class(
            audio_config,
            getattr(audio_config, "samples_per_token", 640),
            config.hidden_size,
        )

    DynamicVLMClass = type(
        "VLM",
        (ParentLLMClass,),  # Inherit from the specific LLM class
        {
            "config_class": config_class,
            "__init__": __init__,
            "init_other_components": init_other_components,
            "_build_connector": _build_connector,
            "_build_audio_connector": _build_audio_connector,
            "_build_vision_model": _build_vision_model,
        },
    )
    return DynamicVLMClass


def install_xmodal_masks(
    self: Any,
    attn2d: Tensor,
    image_block_ids: Tensor | None,
    labels: Tensor | None,
) -> Tensor:
    """Build the cross-modal-arm 4D mask(s) (plan 2026-06-10). Returns the
    tensor to pass downstream as attention_mask; for img2q_window additionally
    stashes the windowed mask on the in-window layers' attention modules
    (consumed by sdpa_xmodal_forward; decode steps shape-guard it away) and
    clears the stash on out-of-window layers. Module-level (like forward) so
    tests can call it unbound with a SimpleNamespace self; attached to the
    dynamic class in the assembly dict below."""
    mode = str(getattr(self.config, "cross_modal_mask_mode", "none") or "none")
    ignore_index = int(getattr(self.config, "ignore_index", -100))
    if mode == "prefix_lm":
        return _xmodal_mask.build_cross_modal_mask(
            attn2d, None, labels, mode, ignore_index=ignore_index
        )
    win = _xmodal_mask.build_cross_modal_mask(
        attn2d, image_block_ids, labels, mode, ignore_index=ignore_index
    )
    base = _xmodal_mask.build_base_mask(attn2d)
    lo, hi = (int(x) for x in getattr(self.config, "cross_modal_mask_window", [1, 9]))
    for idx, layer in enumerate(self.model.layers):
        layer.self_attn._xmodal_mask = win if (lo - 1) <= idx <= (hi - 1) else None
    return base


def _routed_mlp_forward(self: Any, hidden_states: Tensor) -> Tensor:
    """Instance-level FFN forward for a decoder layer carrying a visual expert
    (Mono-InternVL arXiv:2410.08202 / BREEN). Routes each token through the
    text FFN (`_text_mlp_cls_forward`, the original unbound class forward) or
    the visual FFN (`mlp_visual`), blended by the per-token image mask stashed
    on the module by the causal forward. Training always runs BOTH experts (the
    mask is all-zero for text-only batches) so the visual FFN keeps a gradient
    every step — required so DeepSpeed ZeRO does not assert on uneven param
    participation across ranks; with no mask set (cached decode / eval text-only)
    only the text FFN runs. The mask is a non-grad (B, N, 1) tensor that persists
    on the module across the forward+backward of a step, so gradient-checkpoint
    recompute reproduces the same routing."""
    text_out = self._text_mlp_cls_forward(self, hidden_states)
    mask = self._visual_mask
    if mask is None:
        return text_out
    visual_out = self.mlp_visual(hidden_states)
    return text_out * (1.0 - mask) + visual_out * mask


def install_visual_experts(model: Any, config: Any) -> None:
    """Attach a per-decoder-layer modality-routed visual FFN expert to a built
    LM backbone (spec 2026-06-14). The original `layer.mlp` keeps its parameter
    names (so the HF backbone and any checkpoint still load unchanged); we add a
    sibling `layer.mlp.mlp_visual` (a fresh same-class FFN) and override that
    mlp's forward to route by modality. Called from __init__ on BOTH the
    fresh-build and checkpoint-reload paths so the structure exists when weights
    land; the text->visual weight copy is a separate fresh-build-only step
    (init_visual_experts_from_text). Idempotent."""
    if not getattr(config, "visual_expert", False):
        return
    layers_cfg = getattr(config, "visual_expert_layers", None)
    n = len(model.layers)
    idxs = list(range(n)) if not layers_cfg else [int(i) for i in layers_cfg if 0 <= int(i) < n]
    routed: list[nn.Module] = []
    for i in idxs:
        mlp = model.layers[i].mlp
        if getattr(mlp, "_visual_expert_installed", False):
            routed.append(mlp)
            continue
        # Fresh same-class FFN (built under meta during from_pretrained;
        # materialized by _init_weights on fresh build, or by the checkpoint
        # shards on reload — exactly like visual_aux_head).
        mlp.mlp_visual = type(mlp)(config)
        mlp._visual_mask = None
        # The ORIGINAL (text) FFN computation as the unbound class forward:
        # calling it on `self` runs only gate/up/down and never recurses into
        # mlp_visual or this override.
        mlp._text_mlp_cls_forward = type(mlp).forward
        mlp.forward = _routed_mlp_forward.__get__(mlp, type(mlp))
        mlp._visual_expert_installed = True
        routed.append(mlp)
    model._visual_expert_mlps = routed
    log.info(f"Installed visual FFN experts on {len(routed)}/{n} decoder layers.")


def init_visual_experts_from_text(model: Any) -> None:
    """Fresh-build only: copy each layer's text FFN weights into its visual
    expert so the expert starts identical to the text FFN and diverges from
    there (Mono-InternVL init-from-LLM-FFN). No-op when no experts installed."""
    count = 0
    for mlp in getattr(model, "_visual_expert_mlps", []):
        text_sd = {k: v for k, v in mlp.state_dict().items() if not k.startswith("mlp_visual.")}
        mlp.mlp_visual.load_state_dict(text_sd)
        count += 1
    if count:
        log.info(f"Initialized {count} visual FFN experts from their text FFN weights.")


def create_dynamic_causal_vlm_class(
    base_language_model_name_or_path: str,  # e.g., "google/gemma-3-4b-it"
    pretrain_class: Any,
    config_class: Any,
    ParentCausalLLMClass: Any,
):
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
            raise ValueError(f"unknown visual_aux_objective {objective!r} (none|aim_pixel|nepa)")
        depth = int(getattr(config, "visual_aux_head_depth", 2) or 2)
        hidden = int(getattr(config, "visual_aux_head_hidden", 0) or config.hidden_size)
        layers: list[nn.Module] = []
        in_dim = int(config.hidden_size)
        for _ in range(depth - 1):
            layers.extend([nn.Linear(in_dim, hidden), nn.GELU()])
            in_dim = hidden
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    @override
    def forward(
        self: Any,
        input_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: LongTensor | None = None,
        past_key_values: list[FloatTensor] | None = None,
        labels: LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        images: FloatTensor | None = None,
        image_sizes: list[list[int]] | None = None,
        image_position_ids: list[Tensor] | None = None,
        audios: list[Tensor] | None = None,
        return_dict: bool | None = None,
    ) -> CausalLMOutputWithPast:
        # Early modality encoding - encode all image/audio inputs at the beginning
        image_features = None
        audio_features = None
        if images is not None:
            if self.model.vision_model is None:
                # Encoder-free path: images are per-image raw patch tensors.
                image_features = self.encode_raw_patches(images, image_position_ids)
            else:
                image_features, _ = self.encode_images(images)
        if audios is not None:
            audio_features = self.encode_raw_audio(audios)
        # Visual-aux gating (spec 2026-06-06): block ids are built only when
        # the head exists, λ > 0, and we are on the training-loss path.
        visual_aux_on = (
            self.training
            and labels is not None
            and getattr(self, "visual_aux_head", None) is not None
            and float(getattr(self.config, "visual_aux_weight", 0.0) or 0.0) > 0.0
        )
        xmodal_mode = str(getattr(self.config, "cross_modal_mask_mode", "none") or "none")
        # Visual FFN experts (spec 2026-06-14) need the per-token image mask, so
        # request image_block_ids whenever they are enabled (same source the
        # visual-aux / img2q_window paths use).
        has_ve = bool(getattr(self.config, "visual_expert", False))
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
                with_image_block_ids=visual_aux_on or xmodal_mode == "img2q_window" or has_ve,
            )
        # Cross-modal mask arms (plan 2026-06-10): swap the 2D padding mask
        # for the arm's 4D mask on the loss path; generation prefill installs
        # its mask via the one-shot _xmodal_gen_mask stash set in generate().
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
        # Visual FFN expert routing (spec 2026-06-14): stash the per-token image
        # mask on each expert layer's mlp before the backbone runs. Training
        # always provides a mask (all-zero for text-only batches) so both
        # experts get a gradient every step (DeepSpeed ZeRO uneven-participation
        # guard); eval / cached-decode with no image leaves it None -> text only.
        if has_ve:
            ve_gen = getattr(self, "_ve_gen_mask", None)
            if (
                ve_gen is not None
                and inputs_embeds is not None
                and inputs_embeds.shape[1] == ve_gen.shape[1]
            ):
                # Generation prefill: consume the one-shot mask stashed by
                # generate() (image_block_ids are unavailable here because
                # inputs_embeds is already spliced). Decode steps don't match
                # the prefill length -> fall through to text-only (mask=None).
                vmask = ve_gen.to(inputs_embeds.dtype)
                self._ve_gen_mask = None
            elif image_block_ids is not None:
                vmask = (image_block_ids >= 0).unsqueeze(-1).to(
                    inputs_embeds.dtype if inputs_embeds is not None else self.dtype
                )
            elif self.training and inputs_embeds is not None:
                vmask = inputs_embeds.new_zeros(
                    (inputs_embeds.shape[0], inputs_embeds.shape[1], 1)
                )
            elif self.training and input_ids is not None:
                vmask = torch.zeros(
                    (input_ids.shape[0], input_ids.shape[1], 1),
                    dtype=self.dtype,
                    device=input_ids.device,
                )
            else:
                vmask = None
            self._set_visual_mask(vmask)
        loss_chunk_size = getattr(self.config, "loss_chunk_size", 0) or 0
        if (
            loss_chunk_size > 0
            and labels is not None
            and self.training
            and not output_attentions
            and not output_hidden_states
        ):
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
        return super(self.__class__, self).forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

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
        # consumed by the visual-aux loss (wired in the loss body below)
        images: list[Tensor] | None = None,
        image_features: list[Tensor] | None = None,
        image_block_ids: LongTensor | None = None,
    ) -> CausalLMOutputWithPast:
        """Training-only loss path that never materializes the full
        (batch*seq, vocab) fp32 logits: ignore_index positions are dropped
        BEFORE the lm_head matmul (the image splice and user turns are
        ignore_index — often most of the row), and the survivors go through
        lm_head + fp32 CE in `chunk_size`-token chunks.

        Numerically replicates transformers 5.x ForCausalLMLoss with
        num_items_in_batch=None (loss/loss_utils.py): identical shift (the
        hidden state at i predicts labels[i+1]; the last position's target is
        padded to ignore_index), identical matmul-in-model-dtype -> fp32
        upcast order, and sum/count over valid targets — which equals
        cross_entropy's reduction="mean" with ignore_index. Pinned by
        tests in devtools/test_chunked_ce.py.

        Returns logits=None; the HF Trainer never reads logits while
        training, and the eval/generation paths keep the full-logits
        super().forward (this method is gated on self.training).
        """
        # Aux-exit deep supervision (early-fusion ablation; spec:
        # docs/superpowers/specs/2026-06-05-aux-exit-loss-design.md). Layer-k
        # outputs are captured with scoped forward hooks rather than
        # output_hidden_states: the 5.x capture machinery resolves
        # _CAN_RECORD_REGISTRY by str(self.__class__), which is not guaranteed
        # for this dynamically generated backbone class, and a scoped hook
        # also only keeps the layers we need.
        aux_layers = sorted(
            {int(k) for k in (getattr(self.config, "aux_exit_layers", None) or [])}
        )
        aux_weight = float(getattr(self.config, "aux_exit_weight", 0.0) or 0.0)
        aux_active = bool(aux_layers) and aux_weight > 0.0
        # Visual-aux loss (spec 2026-06-06): forward only passes
        # image_block_ids when the head exists, λ > 0 and we're training,
        # so its presence is the single activation signal here.
        va_objective = str(getattr(self.config, "visual_aux_objective", "none") or "none")
        va_weight = float(getattr(self.config, "visual_aux_weight", 0.0) or 0.0)
        va_layer = getattr(self.config, "visual_aux_layer", None)
        va_active = image_block_ids is not None and va_objective != "none" and va_weight > 0.0
        captured: dict[int, Tensor] = {}
        handles = []
        if aux_active:
            num_layers = len(self.model.layers)
            bad = [k for k in aux_layers if not 1 <= k <= num_layers - 1]
            if bad:
                raise ValueError(
                    f"aux_exit_layers {bad} out of range [1, {num_layers - 1}]: k is "
                    f"the 1-based decoder layer whose output feeds the exit, and "
                    f"k == {num_layers} would duplicate the main loss"
                )

        def _make_capture(k: int):
            def _capture(_mod: Any, _args: Any, out: Any) -> None:
                captured[k] = out[0] if isinstance(out, tuple) else out

            return _capture

        capture_layers = sorted(
            set(aux_layers if aux_active else [])
            | ({int(va_layer)} if (va_active and va_layer) else set())
        )
        for k in capture_layers:
            handles.append(self.model.layers[k - 1].register_forward_hook(_make_capture(k)))
        try:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
            )
        finally:
            # Removed before backward ever runs, so gradient-checkpointing
            # recomputation (which re-executes layer __call__ during backward)
            # can never re-fire these hooks.
            for h in handles:
                h.remove()
        hidden_states = outputs.last_hidden_state
        # ForCausalLMLoss's shift: pad labels right with ignore_index, drop
        # the first -> target[i] = labels[i+1], last target ignored. -100 is
        # the fixed ignore_index of the reference implementation.
        shift_targets = nn.functional.pad(labels, (0, 1), value=-100)[..., 1:]
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_targets = shift_targets.reshape(-1).to(flat_hidden.device)
        valid = flat_targets != -100
        n_valid = int(valid.sum())
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

    @override
    def generate(
        self: Any,
        inputs: Tensor | None = None,
        images: FloatTensor | None = None,
        image_sizes: list[list[int]] | None = None,
        image_position_ids: list[Tensor] | None = None,
        audios: list[Tensor] | None = None,
        **kwargs: Any,
    ) -> Any:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None or audios is not None:
            image_features = None
            audio_features = None
            if images is not None:
                if self.model.vision_model is None:
                    image_features = self.encode_raw_patches(images, image_position_ids)
                else:
                    image_features, _ = self.encode_images(images)
            if audios is not None:
                audio_features = self.encode_raw_audio(audios)
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
                    with_image_block_ids=(xmodal_mode == "img2q_window")
                    or bool(getattr(self.config, "visual_expert", False)),
                )
            )
            if xmodal_mode != "none" and attention_mask is not None:
                # Prefill-only custom mask; whole prompt = prefix (labels=None).
                # forward() swaps it in via the shape-matched one-shot stash;
                # decode steps are plain causal text rows in both arms. v1
                # installs the custom mask only on the IMAGE path (this branch):
                # the experiment is about image arms, lmms-eval vision tasks
                # always carry an image, and text-only prompts keep stock causal.
                self._xmodal_gen_mask = self.install_xmodal_masks(
                    attention_mask, image_block_ids, None
                )
            # Visual FFN experts (spec 2026-06-14): stash a one-shot prefill
            # image mask so generation routes image tokens through the expert
            # (forward consumes it by shape-match, like _xmodal_gen_mask). Decode
            # steps emit text tokens -> text FFN, correct by construction.
            if bool(getattr(self.config, "visual_expert", False)) and image_block_ids is not None:
                self._ve_gen_mask = (image_block_ids >= 0).unsqueeze(-1).to(inputs_embeds.dtype)
        else:
            inputs_embeds = self.get_input_embeddings()(inputs)

        return super(self.__class__, self).generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )

    @override
    def prepare_inputs_for_generation(
        self: Any,
        input_ids: Tensor,
        past_key_values: list[FloatTensor] | None = None,
        inputs_embeds: Tensor | None = None,
        **kwargs: Any,
    ):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        image_position_ids = kwargs.pop("image_position_ids", None)
        audios = kwargs.pop("audios", None)
        inputs = super(self.__class__, self).prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        # v5: the base prepare_inputs_for_generation only injects `cache_position`
        # for remote-code models, so it may be absent here. Pop with a default to
        # avoid KeyError; the VLM forward generates from inputs_embeds and does not
        # consume cache_position. Audit point #5.
        inputs.pop("cache_position", None)
        # v5: generate(..., stop_strings=..., tokenizer=...) leaks the tokenizer
        # into model_kwargs (generation/utils._sample has no explicit tokenizer
        # param, so it rides **model_kwargs into forward). Stock models absorb
        # it via forward(**kwargs); our explicit-signature forward must drop it.
        inputs.pop("tokenizer", None)
        inputs.pop("assistant_tokenizer", None)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        if image_position_ids is not None:
            inputs["image_position_ids"] = image_position_ids
        if audios is not None:
            inputs["audios"] = audios
        return inputs

    def encode_raw_patches(
        self: Any,
        images: list[Tensor] | Tensor,
        image_position_ids: list[Tensor] | Tensor | None,
    ) -> list[Tensor]:
        """Encoder-free image path (gemma4_unified-style; vision_model is None).

        `images` are per-image raw patch tensors (N_i, patch_dim) produced by
        RawImageProcessor — N_i varies with aspect ratio — with matching
        (N_i, 2) integer XY coordinates in `image_position_ids` (the spatial
        information lives ONLY in these coordinates; the LM uses plain 1D RoPE).
        All images are packed into one connector call and split back, so the
        cost is a single matmul regardless of batch composition.

        Returns a list of per-image features (N_i, text_hidden) — the same
        per-image layout that prepare_inputs_labels_for_multimodal consumes.
        """
        if isinstance(images, Tensor):
            images = [images]
        if isinstance(image_position_ids, Tensor):
            image_position_ids = [image_position_ids]
        if image_position_ids is None or len(image_position_ids) != len(images):
            raise ValueError(
                "encode_raw_patches: the encoder-free path requires one image_position_ids "
                f"tensor per image, got {0 if image_position_ids is None else len(image_position_ids)} "
                f"for {len(images)} image(s). RawImageProcessor produces them together; make sure "
                "the dataset/collator forwards both lists."
            )
        split_sizes = [patches.shape[0] for patches in images]
        packed = torch.cat([patches.to(self.device) for patches in images], dim=0)
        positions = torch.cat([pos.to(self.device) for pos in image_position_ids], dim=0)
        features = self.model.connector(packed, positions)
        return list(torch.split(features, split_sizes, dim=0))

    def encode_raw_audio(
        self: Any,
        audios: list[Tensor] | Tensor,
    ) -> list[Tensor]:
        """Encoder-free audio path (gemma4_unified-style).

        `audios` are per-audio raw waveform-frame tensors (T_i, samples_per_token)
        from Gemma4UnifiedAudioFeatureExtractor — T_i varies with duration
        (1 frame = 40ms @ 16kHz for the default 640). No position ids: audio is
        a 1D sequence, frame order in the text sequence is all the position
        information there is (gemma4_unified gives audio no positional embedding
        either). Packed into one connector call and split back, symmetric with
        encode_raw_patches.

        Returns a list of per-audio features (T_i, text_hidden).
        """
        if self.model.audio_connector is None:
            raise ValueError(
                "encode_raw_audio: audios were provided but this model has no audio "
                "pathway. Enable it via the model yaml audio section (audio.enabled: true) "
                "so load_model builds the audio connector."
            )
        if isinstance(audios, Tensor):
            audios = [audios]
        split_sizes = [frames.shape[0] for frames in audios]
        packed = torch.cat([frames.to(self.device) for frames in audios], dim=0)
        features = self.model.audio_connector(packed)
        return list(torch.split(features, split_sizes, dim=0))

    def encode_images_raw(self: Any, images: Tensor) -> tuple[list[Tensor] | Tensor, Any]:
        """Encode images using vision model only, without connector."""
        outputs = self.model.vision_model(
            images,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[self.config.vision_config.output_layer].to(
            images.dtype
        )
        if self.config.vision_config.use_all_tokens:
            image_features = hidden_states
        elif self.config.vision_config.use_cls_token:
            if "siglip" in self.config.vision_config.hf_name:
                image_features = outputs.pooler_output.unsqueeze(1)
            else:
                image_features = hidden_states[:, 0:1]
        else:
            image_features = hidden_states[:, 1:]
        image_features = self.model.connector(image_features)
        return image_features, outputs

    def encode_images(
        self: Any,
        images: list[Tensor] | Tensor,
    ):
        if images is None:  # pyright: ignore[reportUnnecessaryComparison]
            image_features = None
            outputs = None
            print("Images are None in this batch, training on text only.")
        elif isinstance(images, list) or images.ndim == 5:
            if isinstance(images, list):
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]  # pyright: ignore
            concat_images = torch.cat([image for image in images], dim=0)  # pyright: ignore
            image_features, outputs = self.encode_images_raw(concat_images)
            split_sizes = [image.shape[0] for image in images]  # pyright: ignore
            image_features: tuple[Tensor, ...] = torch.split(image_features, split_sizes, dim=0)  # pyright: ignore
            image_features = [x.flatten(0, 1) for x in image_features]  # pyright: ignore
        else:
            image_features, outputs = self.encode_images_raw(images)

        return image_features, outputs

    def unpad_image(self: Any, tensor: Tensor, original_size: tuple[int, int]) -> Tensor:
        """
        Unpads a PyTorch tensor of a padded and resized image.

        Args:
        tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
        original_size (tuple): The original size of PIL image (width, height).

        Returns:
        torch.Tensor: The unpadded image tensor.
        """
        original_width, original_height = original_size
        current_height, current_width = tensor.shape[1:]

        original_aspect_ratio = original_width / original_height
        current_aspect_ratio = current_width / current_height

        if original_aspect_ratio > current_aspect_ratio:
            scale_factor = current_width / original_width
            new_height = int(original_height * scale_factor)
            padding = (current_height - new_height) // 2
            unpadded_tensor = tensor[:, padding : current_height - padding, :]
        else:
            scale_factor = current_height / original_height
            new_width = int(original_width * scale_factor)
            padding = (current_width - new_width) // 2
            unpadded_tensor = tensor[:, :, padding : current_width - padding]

        return unpadded_tensor

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
        if (image_features is None and audio_features is None) or input_ids.shape[1] == 1:  # pyright: ignore
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0,
                input_ids.shape[1],  # pyright: ignore
                dtype=torch.long,
                device=input_ids.device,  # pyright: ignore
            )
        if labels is None:
            labels = torch.full_like(input_ids, self.config.ignore_index)  # pyright: ignore

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask, strict=False)
        ]  # pyright: ignore
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask, strict=False)
        ]  # pyright: ignore

        # Modality registry: placeholder token id -> (feature list, cursor).
        # Cursors are shared across the batch loop: features are ordered exactly
        # as their placeholders appear across the batch, plus one dummy entry per
        # sample that lacks the modality (consumed zero-width below — the LLaVA
        # deepspeed trick: the connector already ran on the dummy during the
        # encode phase, so its parameters stay in the graph every step).
        modality_features: dict[int, tuple[Any, list[int]]] = {}
        if image_features is not None:
            modality_features[self.config.image_token_index] = (image_features, [0])
        if audio_features is not None:
            modality_features[getattr(self.config, "audio_token_index", -201)] = (
                audio_features,
                [0],
            )

        new_input_embeds = []
        new_labels = []
        # Visual-aux block-id tracking (spec 2026-06-06 §3.2): which spliced
        # positions belong to which image (flat batch-cursor index). Built only
        # on request so the baseline assembly loop stays byte-identical.
        new_image_block_ids: list[Tensor] | None = [] if with_image_block_ids else None
        for batch_idx, cur_input_ids in enumerate(input_ids):
            is_mm_token = torch.zeros_like(cur_input_ids, dtype=torch.bool)
            for token_index in modality_features:
                is_mm_token |= cur_input_ids == token_index
            mm_positions: list[int] = torch.where(is_mm_token)[0].tolist()

            # Zero-width dummy consumption for modalities absent from this row.
            absent_dummies = []
            for token_index, (features, cursor) in modality_features.items():
                if not bool((cur_input_ids == token_index).any()):
                    absent_dummies.append(features[cursor[0]][0:0])
                    cursor[0] += 1

            if len(mm_positions) == 0:
                cur_input_embeds_1 = self.get_input_embeddings()(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, *absent_dummies], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])  # pyright: ignore
                if new_image_block_ids is not None:
                    new_image_block_ids.append(
                        torch.full_like(labels[batch_idx], -1)  # pyright: ignore
                    )
                continue

            # Split the row into text segments around the (interleaved) modality
            # placeholders, embed all text in one lookup, then re-assemble with
            # each placeholder replaced by its modality's next feature block.
            boundaries = [-1] + mm_positions + [cur_input_ids.shape[0]]
            cur_input_ids_segments = []
            cur_labels = labels[batch_idx]  # pyright: ignore
            cur_labels_segments = []
            for i in range(len(boundaries) - 1):
                cur_input_ids_segments.append(cur_input_ids[boundaries[i] + 1 : boundaries[i + 1]])
                cur_labels_segments.append(cur_labels[boundaries[i] + 1 : boundaries[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_segments]
            cur_input_embeds = self.get_input_embeddings()(torch.cat(cur_input_ids_segments))
            text_segment_embeds = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_new_block_ids: list[Tensor] = []

            for i in range(len(mm_positions) + 1):
                cur_new_input_embeds.append(text_segment_embeds[i])
                cur_new_labels.append(cur_labels_segments[i])
                if new_image_block_ids is not None:
                    cur_new_block_ids.append(torch.full_like(cur_labels_segments[i], -1))
                if i < len(mm_positions):
                    token_index = int(cur_input_ids[mm_positions[i]].item())
                    features, cursor = modality_features[token_index]
                    feature_index = cursor[0]
                    cur_features = features[feature_index]
                    cursor[0] += 1
                    cur_new_input_embeds.append(cur_features)
                    cur_new_labels.append(
                        torch.full(
                            (cur_features.shape[0],),
                            self.config.ignore_index,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )
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

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds.extend(absent_dummies)

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            if new_image_block_ids is not None:
                new_image_block_ids.append(torch.cat(cur_new_block_ids))

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = self.config.max_seq_length
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            if new_image_block_ids is not None:
                new_image_block_ids = [x[:tokenizer_model_max_length] for x in new_image_block_ids]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            self.config.ignore_index,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
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
        attention_mask = torch.zeros(
            (batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device
        )
        position_ids = torch.zeros(
            (batch_size, max_len),
            dtype=position_ids.dtype,  # pyright: ignore
            device=position_ids.device,  # pyright: ignore
        )  # pyright: ignore

        for i, (cur_new_embed, cur_new_labels) in enumerate(
            zip(new_input_embeds, new_labels, strict=False)
        ):
            cur_len = cur_new_embed.shape[0]
            if self.config.padding_side == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    if image_block_ids_padded is not None:
                        image_block_ids_padded[i, -cur_len:] = new_image_block_ids[i]
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(  # pyright: ignore
                        0,
                        cur_len,
                        dtype=position_ids.dtype,  # pyright: ignore
                        device=position_ids.device,  # pyright: ignore
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    if image_block_ids_padded is not None:
                        image_block_ids_padded[i, :cur_len] = new_image_block_ids[i]
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(  # pyright: ignore
                        0,
                        cur_len,
                        dtype=position_ids.dtype,  # pyright: ignore
                        device=position_ids.device,  # pyright: ignore
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return (  # pyright: ignore
            None,
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
            image_block_ids_padded,
        )

    def _set_visual_mask(self: Any, mask: Tensor | None) -> None:
        """Stash the per-token image mask (B, N, 1) on every visual-expert mlp
        so _routed_mlp_forward can blend text/visual FFN outputs. No-op unless
        experts are installed (install_visual_experts populated the list)."""
        mlps = getattr(self.model, "_visual_expert_mlps", None)
        if mlps:
            for mlp in mlps:
                mlp._visual_mask = mask

    def floating_point_ops(
        self: Any, input_dict: dict[str, Any], exclude_embeddings: bool = True
    ) -> int:
        """Trainer-side FLOPs estimate (6 * params * tokens), accumulated into
        trainer_state.total_flos. Overridden because the default counts
        input_ids tokens, where each image is a single sentinel — the spliced
        sequence the model actually runs has up to max_soft_tokens (280) per
        image. Swap every media sentinel for its true expansion: soft-token
        counts come from image_position_ids lengths / audio frame counts."""
        input_ids = input_dict.get("input_ids")
        if input_ids is None:
            return 0
        mask = input_dict.get("attention_mask")
        tokens = int(mask.sum().item()) if mask is not None else input_ids.numel()
        image_position_ids = input_dict.get("image_position_ids")
        if image_position_ids:
            tokens += sum(int(p.shape[0]) for p in image_position_ids)
            image_token_index = getattr(self.config, "image_token_index", -200)
            n_sentinels = int((input_ids == image_token_index).sum().item())
            tokens -= n_sentinels  # each sentinel is replaced by its expansion
            # Absent-modality dummies carry one placeholder row (shape (1, 2))
            # but splice in zero-width and have no sentinel — don't count them.
            tokens -= max(0, len(image_position_ids) - n_sentinels)
        elif input_dict.get("images") is not None:
            # Classic encoder path (CLIP/SigLIP/DINO): no per-image position
            # ids; every real image (one sentinel each) splices a fixed
            # soft-token count derived from the tower config — keep total_flos
            # comparable with the encoder-free arm.
            image_token_index = getattr(self.config, "image_token_index", -200)
            n_sentinels = int((input_ids == image_token_index).sum().item())
            vision_config = getattr(self.config, "vision_config", None)
            if n_sentinels and vision_config is not None:
                get = (
                    vision_config.get
                    if isinstance(vision_config, dict)
                    else lambda k: getattr(vision_config, k, None)
                )
                image_size, patch_size = get("image_size"), get("patch_size")
                if image_size and patch_size:
                    soft = (image_size // patch_size) ** 2
                    if getattr(self.config, "use_cls_token", False):
                        soft += 1
                    tokens += n_sentinels * (soft - 1)  # sentinel itself already counted
        audios = input_dict.get("audios")
        if audios:
            tokens += sum(int(a.shape[0]) for a in audios)
            audio_token_index = getattr(self.config, "audio_token_index", -201)
            n_sentinels = int((input_ids == audio_token_index).sum().item())
            tokens -= n_sentinels
            tokens -= max(0, len(audios) - n_sentinels)  # zero-width dummies
        return 6 * tokens * self.num_parameters(exclude_embeddings=exclude_embeddings)

    DynamicCausalVLMClass = type(
        "VLMForCausalLM",
        (ParentCausalLLMClass,),  # Inherit from the specific LLM class
        {
            "config_class": config_class,
            "__init__": __init__,
            "_build_visual_aux_head": _build_visual_aux_head,
            "_set_visual_mask": _set_visual_mask,
            "forward": forward,
            "install_xmodal_masks": install_xmodal_masks,
            "chunked_ce_forward": chunked_ce_forward,
            "floating_point_ops": floating_point_ops,
            "generate": generate,
            "prepare_inputs_for_generation": prepare_inputs_for_generation,
            "encode_raw_patches": encode_raw_patches,
            "encode_raw_audio": encode_raw_audio,
            "encode_images_raw": encode_images_raw,
            "encode_images": encode_images,
            "unpad_image": unpad_image,
            "prepare_inputs_labels_for_multimodal": prepare_inputs_labels_for_multimodal,
        },
    )

    return DynamicCausalVLMClass


def get_dynamic_vlm(
    base_language_model_name_or_path: str,
):
    ParentLLMClass, ParentCausalLLMClass, base_language_model_name_or_path = get_dynamic_vlm_class(
        base_language_model_name_or_path
    )
    VLMConfig = create_dynamic_vlm_config_class(base_language_model_name_or_path)
    VLM = create_dynamic_vlm_class(base_language_model_name_or_path, VLMConfig, ParentLLMClass)
    VLMForCausalLM = create_dynamic_causal_vlm_class(
        base_language_model_name_or_path,
        VLM,
        VLMConfig,
        ParentCausalLLMClass,
    )
    return VLMForCausalLM, VLMConfig
