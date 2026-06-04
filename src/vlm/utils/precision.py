import logging

import torch
import torch.distributed as dist
from transformers.utils import (
    is_torch_bf16_gpu_available,
    is_torch_tf32_available,
)

log: logging.Logger = logging.getLogger(__name__)


def _dist_min_bool(value: bool) -> bool:
    """If distributed is initialized, compute logical AND across ranks via MIN.

    Returns the same value if not in a distributed context.
    """
    if not dist.is_available() or not dist.is_initialized():
        return value
    try:
        t = torch.tensor(
            [1 if value else 0], device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        return bool(t.item())
    except Exception:  # best-effort; never crash resolution
        return value


def resolve_precision(
    bf16: bool | None,
    tf32: bool | None,
) -> tuple[bool, bool]:
    """Resolve final bf16/tf32 according to the policy:
    - If user provided True/False (not None), honor user choice (even if unsupported), log a warning if conflicting.
    - If None, auto-detect support; in distributed, require all ranks to support (logical AND).
    - After resolving, set torch backends for tf32 accordingly.
    Returns (use_bf16, use_tf32).
    """
    # Detect capabilities locally
    bf16_supported = is_torch_bf16_gpu_available()
    tf32_supported = is_torch_tf32_available()

    # Sync detection across ranks for auto mode
    bf16_supported_all = _dist_min_bool(bf16_supported)
    tf32_supported_all = _dist_min_bool(tf32_supported)

    # Resolve bf16
    if bf16 is None:
        use_bf16 = bf16_supported_all
        log.info(
            f"bf16=auto -> resolved to {use_bf16} (local_supported={bf16_supported}, all_ranks_supported={bf16_supported_all})"
        )
    else:
        use_bf16 = bf16
        if bf16 and not bf16_supported_all:
            log.warning(
                "bf16 requested by user but appears unsupported on some/all ranks. Proceeding as requested; this may error at runtime."
            )

    # Resolve tf32
    if tf32 is None:
        use_tf32 = tf32_supported_all
        log.info(
            f"tf32=auto -> resolved to {use_tf32} (local_supported={tf32_supported}, all_ranks_supported={tf32_supported_all})"
        )
    else:
        use_tf32 = tf32
        if tf32 and not tf32_supported_all:
            log.warning(
                "tf32 requested by user but appears unsupported on some/all ranks. Proceeding as requested; library may ignore this setting."
            )

    # Apply tf32 backend switches immediately
    try:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = use_tf32  # type: ignore[attr-defined]
            torch.backends.cudnn.allow_tf32 = use_tf32  # type: ignore[attr-defined]
    except Exception:
        pass

    return use_bf16, use_tf32
