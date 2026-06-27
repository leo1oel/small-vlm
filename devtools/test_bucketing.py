"""End-to-end check of length-bucketed batching against the real Azure stream
(CPU only — runs on a login node; needs AZURE_SAS_TOKEN via .env).

Builds the energon loader twice (plain vs bucketed task encoder), pulls a few
batches from each, and reports:
  - batch fullness and bucket-bound compliance (bucketed loader),
  - pad-to-batch-max waste on the effective (post-splice) lengths,
  - that the same collator contract (keys/shapes) holds in both modes.

Usage:  .venv/bin/python devtools/test_bucketing.py
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hydra import compose, initialize_config_dir  # noqa: E402

from vlm.config import register_configs  # noqa: E402
from vlm.data import get_data_args  # noqa: E402
from vlm.data.energon_dataset import (  # noqa: E402
    VLMBucketedChatTaskEncoder,
    VLMChatTaskEncoder,
    build_energon_train_loader,
    effective_sample_length,
)
from vlm.utils import conversation as conversation_lib  # noqa: E402

N_BATCHES = 12
BS = 4


def build_processor(cfg):
    from transformers import AutoTokenizer

    from vlm.models import VLMProcessor
    from vlm.models.image_processing_raw import RawImageProcessor

    ip = RawImageProcessor(
        patch_size=cfg.model.visual_encoder.patch_size,
        pooling_kernel_size=cfg.model.visual_encoder.pooling_kernel_size,
        max_soft_tokens=cfg.model.visual_encoder.max_soft_tokens,
    )
    tok = AutoTokenizer.from_pretrained(
        cfg.model.language_model.hf_name,
        use_fast=True,
        model_max_length=cfg.model.language_model.max_seq_length,
        padding_side=cfg.model.language_model.padding_side,
    )
    return VLMProcessor(image_processor=ip, tokenizer=tok)


class _Recorder:
    """Mixin factory: wraps a task encoder class so batch() records the
    effective lengths of the samples it collates."""

    @staticmethod
    def wrap(encoder_cls):
        class Recorded(encoder_cls):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.recorded: list[list[int]] = []

            def batch(self, samples):
                self.recorded.append([effective_sample_length(s, self.data_args) for s in samples])
                return super().batch(samples)

        return Recorded


def run(cfg, data_args, processor, bucketed: bool):
    edges = list(cfg.dataset.length_buckets) if bucketed else None
    if bucketed:
        enc = _Recorder.wrap(VLMBucketedChatTaskEncoder)(processor, data_args, edges)
    else:
        enc = _Recorder.wrap(VLMChatTaskEncoder)(processor, data_args)
    loader = build_energon_train_loader(
        cfg.dataset, processor, data_args, batch_size=BS, task_encoder=enc
    )
    batches = []
    for i, b in enumerate(loader):
        assert "input_ids" in b and "labels" in b, f"collator contract broken: {b.keys()}"
        batches.append(b)
        if i + 1 >= N_BATCHES:
            break
    waste_tok = total_tok = 0
    full = 0
    for lens in enc.recorded[:N_BATCHES]:
        mx = max(lens)
        total_tok += len(lens) * mx
        waste_tok += len(lens) * mx - sum(lens)
        full += len(lens) == BS
        if bucketed:
            import bisect

            keys = {bisect.bisect_left(edges, n) for n in lens}
            assert len(keys) == 1, f"batch mixes buckets: lens={lens} keys={keys}"
    print(
        f"{'bucketed' if bucketed else 'plain   '}: batches={len(enc.recorded[:N_BATCHES])} "
        f"full={full}/{N_BATCHES} padding_waste={100 * waste_tok / total_tok:.1f}%",
        flush=True,
    )
    return waste_tok / total_tok


def main():
    register_configs()
    with initialize_config_dir(config_dir=str(REPO / "src/vlm/config"), version_base=None):
        cfg = compose(
            config_name="sft-unified",
            overrides=[
                # workers=0: batching runs inline in this process, so the
                # recording wrapper actually observes it (workers would fork
                # and keep their recordings to themselves).
                "dataset.num_workers=0",
                "dataset.shuffle_buffer_size=50",
            ],
        )
    conversation_lib.default_conversation = conversation_lib.conv_templates[cfg.trainer.version]
    data_args = get_data_args(cfg.dataset, cfg.model)
    processor = build_processor(cfg)

    print(f"pulling {N_BATCHES} batches x bs{BS} from the live stream, twice...", flush=True)
    w_plain = run(cfg, data_args, processor, bucketed=False)
    w_bucket = run(cfg, data_args, processor, bucketed=True)
    assert w_bucket < w_plain, "bucketing did not reduce padding waste"
    print(f"OK: padding waste {100 * w_plain:.1f}% -> {100 * w_bucket:.1f}%", flush=True)


if __name__ == "__main__":
    main()
