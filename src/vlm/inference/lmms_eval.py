"""lmms-eval (>= 0.7) chat-model adapter for small-vlm checkpoints.

Bridges the lmms-eval harness onto the production inference path
(vlm.inference.generate_response), so benchmark prompts get exactly the
training-aligned template/preprocessing for both model families
(encoder-free unified and legacy CLIP/SigLIP).

lmms-eval 0.7 resolves ``--model`` through ModelRegistryV2 (class-path
manifests), NOT the old @register_model decorator registry — external
models must register a manifest before cli_evaluate runs. Use the
launcher::

    python devtools/run_lmms_eval.py --model small-vlm \\
        --model_args pretrained=outputs/sft-unified/checkpoint-3000 \\
        --tasks mme --batch_size 1 --output_path logs/lmms_eval

Extra ``--model_args`` keys: conv_mode (template override),
image_aspect_ratio (legacy encoder checkpoints that predate the config
recording: pass square/pad), bf16/fp16, attn_implementation, device.
"""

import logging
from typing import Any

from tqdm import tqdm

from .eval import _data_args_from_config, generate_response, load_model, resolve_conv_mode

log: logging.Logger = logging.getLogger(name=__name__)

try:
    from lmms_eval.api.instance import Instance
    from lmms_eval.api.model import lmms
    from lmms_eval.protocol import ChatMessages
except ImportError as e:  # pragma: no cover - eval extra not installed
    raise ImportError(
        "lmms-eval is required for the benchmark adapter: uv pip install lmms-eval"
    ) from e


def register() -> None:
    """Register 'small-vlm' with lmms-eval's ModelRegistryV2 (idempotent)."""
    from lmms_eval.models import MODEL_REGISTRY_V2
    from lmms_eval.models.registry_v2 import ModelManifest

    MODEL_REGISTRY_V2.register_manifest(
        ModelManifest(
            model_id="small-vlm",
            chat_class_path="vlm.inference.lmms_eval.SmallVLM",
        ),
        overwrite=True,
    )


class SmallVLM(lmms):
    """Chat-interface adapter: one request at a time through generate_response."""

    is_simple = False  # chat model: requests carry doc_to_messages

    def __init__(
        self,
        pretrained: str,
        device: str | None = None,
        batch_size: int | str = 1,
        bf16: bool = True,
        fp16: bool = False,
        attn_implementation: str = "sdpa",
        conv_mode: str | None = None,
        image_aspect_ratio: str | None = None,
        max_new_tokens: int = 1024,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if kwargs:
            log.warning("SmallVLM: ignoring unexpected model_args %s", sorted(kwargs))
        if int(batch_size) != 1:
            raise ValueError("small-vlm adapter runs batch_size=1 (single-sample generate)")

        # The harness may chdir during evaluation — pin local checkpoint paths
        # down before anything resolves them (a dangling relative path would be
        # misread as a hub repo id by transformers' loaders).
        from pathlib import Path

        if Path(pretrained).exists():
            pretrained = str(Path(pretrained).resolve())

        model, processor, info = load_model(
            pretrained, bf16=bf16, fp16=fp16, attn_implementation=attn_implementation, device=device
        )
        self._model = model
        self.processor = processor
        self.conv_mode: str = resolve_conv_mode(
            model.config, pretrained=pretrained, conv_mode=conv_mode
        )
        # Media placeholders must match what the checkpoint was trained with —
        # read them from the model config instead of hardcoding <image>/<audio>,
        # so checkpoints with custom media tokens splice correctly.
        data_args = _data_args_from_config(model.config)
        self._image_token = data_args.image_token
        self._audio_token = data_args.audio_token
        self.image_aspect_ratio = image_aspect_ratio
        self._max_new_tokens = max_new_tokens
        log.info(
            "SmallVLM ready: %s (encoder_free=%s, conv=%s, aspect=%s)",
            pretrained,
            info["encoder_free"],
            self.conv_mode,
            image_aspect_ratio or "from-config",
        )

    def _messages_to_query_and_media(self, chat: ChatMessages) -> tuple[str, list, list]:
        """Flatten chat messages into a single user query with media
        placeholders at their content positions — the same item-to-text rule
        as training's messages_to_conversations ('\\n'-joined typed items)."""
        images, videos, audios = chat.extract_media()
        if videos:
            raise NotImplementedError("small-vlm has no video pathway")
        parts: list[str] = []
        for message in chat.messages:
            if message.role != "user":
                # benchmarks are single-turn; the system prompt is owned by the
                # training template (build_prompt), not the task.
                continue
            for content in message.content:
                if content.type == "text":
                    parts.append(content.text)
                elif content.type == "image":
                    parts.append(self._image_token)
                elif content.type == "audio":
                    parts.append(self._audio_token)
        return "\n".join(parts), images, audios

    def generate_until(self, requests: list[Instance]) -> list[str]:
        results: list[str] = []
        for request in tqdm(requests, disable=(self.rank != 0), desc="small-vlm responding"):
            ctx, doc_to_messages, gen_kwargs, doc_id, task, split = request.args
            doc = self.task_dict[task][split][doc_id]
            chat = ChatMessages(messages=doc_to_messages(doc))
            query, images, audios = self._messages_to_query_and_media(chat)

            gen_kwargs = dict(gen_kwargs or {})
            until = gen_kwargs.pop("until", None)
            if isinstance(until, str):
                until = [until]
            # Thread generation kwargs through explicitly; do_sample is honoured
            # independently of temperature (generate_response derives it from
            # temperature only when do_sample is None), so a task can request
            # sampling at the model's default temperature.
            do_sample = gen_kwargs.get("do_sample", None)
            temperature = float(gen_kwargs.get("temperature") or 0.0)

            text = generate_response(
                self._model,
                self.processor,
                query=query,
                images=images or None,
                audios=audios or None,
                conv_mode=self.conv_mode,
                temperature=temperature,
                top_p=float(gen_kwargs.get("top_p") or 1.0),
                num_beams=int(gen_kwargs.get("num_beams") or 1),
                max_new_tokens=int(gen_kwargs.get("max_new_tokens") or self._max_new_tokens),
                stop_strings=list(until) if until else None,
                image_aspect_ratio=self.image_aspect_ratio,
                do_sample=None if do_sample is None else bool(do_sample),
            )
            results.append(text)
            # Cache key must carry doc/media identity: two docs with identical
            # context text but different images would otherwise collide on
            # (ctx, gen_kwargs). Include (task, split, doc_id) so repeated text
            # with different media is cached separately.
            self.cache_hook.add_partial(
                "generate_until", (ctx, gen_kwargs, task, split, doc_id), text
            )
        return results

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        raise NotImplementedError(
            "small-vlm adapter currently supports generate_until tasks only "
            "(MME/GQA/TextVQA/...); loglikelihood-based MCQ tasks need a "
            "forward-pass scorer — open an issue if needed"
        )

    def generate_until_multi_round(self, requests: list[Instance]) -> list[str]:
        raise NotImplementedError("multi-round generation not implemented for small-vlm")
