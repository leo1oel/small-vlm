import torch
from deepspeed.runtime.fp16.loss_scaler import LossScaler
from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from torch.serialization import safe_globals


def convert_checkpoint():
    original_torch_load = torch.load  # pyright: ignore

    def patched_torch_load(*args, **kwargs):  # pyright: ignore
        kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)  # pyright: ignore

    torch.load = patched_torch_load

    try:
        result = convert_zero_checkpoint_to_fp32_state_dict(
            checkpoint_dir="/pasteur2/u/yuhuiz/yiming/small-vlm/outputs/2025-04-12/00-33-13/checkpoints/last.ckpt",
            output_file="/pasteur2/u/yuhuiz/yiming/small-vlm/outputs/2025-04-12/00-33-13/checkpoints/model.pt",
        )
        print("Successfully converted!")
        return result
    finally:
        torch.load = original_torch_load


if __name__ == "__main__":
    with safe_globals([LossScaler]):
        convert_checkpoint()
