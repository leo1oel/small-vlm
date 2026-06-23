"""Launcher: register the small-vlm adapter, then hand over to lmms-eval's CLI.

lmms-eval resolves --model via ModelRegistryV2, which only knows its bundled
models; the adapter manifest must be registered in-process first.

Usage (same flags as `python -m lmms_eval`):

    python devtools/run_lmms_eval.py --model small-vlm \\
        --model_args pretrained=outputs/sft-unified/checkpoint-3000 \\
        --tasks mme --batch_size 1 --log_samples --output_path logs/lmms_eval
"""

from vlm.inference.lmms_eval import register

register()

from lmms_eval.__main__ import cli_evaluate  # noqa: E402

if __name__ == "__main__":
    cli_evaluate()
