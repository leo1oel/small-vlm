"""Gradient-starvation probe (env-gated, off by default).

Tests the documented multimodal-collapse mechanism (Gradient Starvation,
Pezeshki et al. NeurIPS 2021; modality competition, Wu/Huang ICML 2022) DIRECTLY
on this native VLM: at each optimizer step (pre-step, gradients still live) it
logs the per-parameter RMS gradient magnitude of the VISUAL pathway vs the
LANGUAGE pathway. If the visual RMS collapses relative to language early in
training (and loss saturates ~step 500), the visual gradient is being starved.

Pathway split by parameter name:
  visual   = '.mlp_visual.' (per-layer visual FFN expert) or 'connector'
             (raw-patch projection / patch embedding)
  language = everything else (text FFN gate/up/down, self_attn, embed, norm, lm_head)

RMS (sqrt(sum_sq / numel)) is used, not raw norm, so the comparison is fair
despite the two groups having different parameter counts. Enabled only when
env GRAD_PROBE=1; reads GRAD_PROBE_EVERY (default 5). Single-GPU / no-ZeRO only
(ZeRO-2 shards grads, making p.grad incomplete) — the probe asserts world_size
context is fine because it reads local p.grad which is complete under DDP/single.
"""

import math

from transformers import TrainerCallback


def _group(name: str) -> str:
    """connector = raw-patch projection (RANDOM init — most exposed to
    starvation); vexpert = per-layer visual FFN (init from text FFN); language =
    rest of the LLM backbone."""
    if "connector" in name:
        return "connector"
    if ".mlp_visual." in name:
        return "vexpert"
    return "language"


class GradProbeCallback(TrainerCallback):
    def __init__(self, model, every: int = 5):
        self.model = model
        self.every = max(1, int(every))
        self._header = False

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if step % self.every != 0:
            return
        sq = {"connector": 0.0, "vexpert": 0.0, "language": 0.0}
        ne = {"connector": 0, "vexpert": 0, "language": 0}
        for name, p in self.model.named_parameters():
            if p.grad is None or not p.requires_grad:
                continue
            g = p.grad.detach().float()
            k = _group(name)
            sq[k] += float(g.pow(2).sum().item())
            ne[k] += int(g.numel())

        def rms(k):
            return math.sqrt(sq[k] / ne[k]) if ne[k] else float("nan")
        c, v, l = rms("connector"), rms("vexpert"), rms("language")
        cr = c / l if l > 0 else float("nan")
        vr = v / l if l > 0 else float("nan")
        if not self._header:
            print("[gradprobe] columns: step connector_rms vexpert_rms language_rms "
                  "conn/lang vexp/lang  (RMS = per-param grad magnitude)", flush=True)
            self._header = True
        print(f"[gradprobe] {step} {c:.4e} {v:.4e} {l:.4e} {cr:.4e} {vr:.4e}", flush=True)
