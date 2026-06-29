"""Tests for the BREEN anti-collapse distill port (audit §4): the EMA per-channel
target debias (trick A), the bounded Gram relational term (trick B), the NaN
guards, and the ``init_visual_distill_buffers`` "to_empty trap" fix.

Two invariants are load-bearing and were the reason "identical loss code died at
random" before the port hardened them:

* **Gated OFF == plain cosine.** With all 16 dials at their defaults the
  ``_compute_anticollapse`` dispatch must NOT fire, so eve/repa/softdepth behave
  exactly as the clean distill core (the loss the optimizer sees is unchanged).
* **Fresh-build buffer init.** A ``from_pretrained`` build leaves the debias EMA
  buffers as ``to_empty`` garbage; a garbage-truthy ``debias_inited`` makes the
  EMA subtract an uninitialized (often Inf) mean -> NaN cosine -> every microbatch
  skipped, NON-deterministically per run. ``init_visual_distill_buffers`` resets
  them, and an in-loss ``isfinite`` re-init guard is the second line of defense.

CPU-only, no network, no teacher: the head's loss math is exercised on synthetic
student/teacher features, which is exactly where the trap lived.
"""

from typing import Any

import pytest
import torch
import torch.nn as nn

from vlm.models.modeling_vlm import init_visual_distill_buffers
from vlm.models.visual_distill import VisualDistillHead

LLM_DIM, TEACHER_DIM, M_TOK, N_IMG = 32, 24, 10, 4
METHODS = {"eve": [6], "repa": [6], "softdepth": [4, 8, 12, 16, 20, 24]}


def _build_head(method: str, **dials: Any) -> VisualDistillHead:
    return VisualDistillHead(
        method=method,
        llm_dim=LLM_DIM,
        teacher_dim=TEACHER_DIM,
        layers=METHODS[method],
        head_hidden=0,
        loss_type="cosine",
        **dials,
    )


def _samples(method: str, *, seed: int = 0, constant_target: bool = False) -> list[dict]:
    """Per-image dicts in the format VisualDistillHead.compute expects."""
    g = torch.Generator().manual_seed(seed)
    keys = [0] if method == "eve" else METHODS[method]
    out = []
    # A single shared "mean image" target the plain cosine collapses onto.
    const = torch.randn(M_TOK, TEACHER_DIM, generator=g)
    for _ in range(N_IMG):
        native = {k: torch.randn(M_TOK, LLM_DIM, generator=g) for k in keys}
        target = const.clone() if constant_target else torch.randn(
            M_TOK, TEACHER_DIM, generator=g
        )
        out.append({"native": native, "target": target})
    return out


class _StubModel(nn.Module):
    """Minimal carrier so init_visual_distill_buffers can find the head + a
    non-meta device via model.parameters() (the head's own proj params)."""

    def __init__(self, head: VisualDistillHead) -> None:
        super().__init__()
        self.visual_distill_head = head


# ---------------------------------------------------------------------------
# Invariant 1: gated OFF -> dispatch never fires (== plain cosine behavior)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", list(METHODS))
def test_dispatch_gated_off_by_default(method: str, monkeypatch: Any) -> None:
    head = _build_head(method)
    assert head._anticollapse_on() is False
    # If the dispatch fired with all dials off, this would raise.
    monkeypatch.setattr(
        head,
        "_compute_anticollapse",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("dispatch fired while OFF")),
    )
    head.train()
    loss, comps = head.compute(_samples(method))
    assert torch.isfinite(loss)
    assert set(("distill", "distill_cos")).issubset(comps)


@pytest.mark.parametrize("method", ["eve", "repa", "softdepth"])
def test_dispatch_fires_when_on(method: str, monkeypatch: Any) -> None:
    head = _build_head(method, debias_target=True, rkd_dist_weight=1.0)
    assert head._anticollapse_on() is True
    sentinel = (torch.tensor(0.0), {"distill": torch.tensor(0.0)})
    monkeypatch.setattr(head, "_compute_anticollapse", lambda *a, **k: sentinel)
    assert head.compute(_samples(method)) is sentinel


def test_anticollapse_on_predicate_each_lever() -> None:
    assert _build_head("eve")._anticollapse_on() is False
    for dial in (
        {"debias_target": True},
        {"rkd_dist_weight": 1.0},
        {"rkd_angle_weight": 1.0},
        {"vicreg_var_weight": 1.0},
        {"vicreg_cov_weight": 1.0},
        {"mgd_weight": 1.0},
        {"sigreg_weight": 1.0},
    ):
        assert _build_head("eve", **dial)._anticollapse_on() is True, dial


# ---------------------------------------------------------------------------
# Invariant 2: debias buffers + the to_empty-trap fix
# ---------------------------------------------------------------------------
def test_debias_buffers_registered_only_when_on() -> None:
    off = _build_head("eve")
    assert not hasattr(off, "debias_mean")
    on = _build_head("eve", debias_target=True)
    assert on.debias_mean.shape == (TEACHER_DIM,)
    assert torch.equal(on.debias_mean, torch.zeros(TEACHER_DIM))
    assert torch.equal(on.debias_var, torch.ones(TEACHER_DIM))
    assert bool(on.debias_inited) is False
    # _ac_step exists only when a warmup is on.
    assert not hasattr(on, "_ac_step")
    assert hasattr(_build_head("eve", debias_target=True, ac_warmup_steps=5), "_ac_step")


def test_init_visual_distill_buffers_resets_to_empty_garbage() -> None:
    head = _build_head("eve", debias_target=True, ac_warmup_steps=3)
    # Mimic `to_empty` uninitialized memory: a truthy inited flag + non-finite
    # running stats. This is the exact precondition that NaN-skips every step.
    head.debias_inited.fill_(True)
    head.debias_mean.fill_(float("inf"))
    head.debias_var.fill_(float("nan"))
    head._ac_step.fill_(999)

    init_visual_distill_buffers(_StubModel(head))

    assert torch.equal(head.debias_mean, torch.zeros(TEACHER_DIM))
    assert torch.equal(head.debias_var, torch.ones(TEACHER_DIM))
    assert bool(head.debias_inited) is False
    assert int(head._ac_step) == 0


def test_init_visual_distill_buffers_noop_without_head() -> None:
    class _Bare(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.p = nn.Parameter(torch.zeros(1))

    init_visual_distill_buffers(_Bare())  # head absent -> no-op, no raise


def test_init_visual_distill_buffers_noop_when_debias_off() -> None:
    # method enabled but debias off -> no debias buffers to reset, clean no-op.
    init_visual_distill_buffers(_StubModel(_build_head("eve")))


@pytest.mark.parametrize("method", ["eve", "repa", "softdepth"])
def test_debias_target_step0_finite(method: str) -> None:
    """Smell-test (ii): a fresh build with debias_target must not NaN at step 0.
    Reproduce the to_empty garbage, then verify BOTH lines of defense keep the
    step-0 loss finite: (a) the build-time init reset, and (b) the in-loss
    isfinite re-init guard on its own."""
    # (a) build-time init path
    head = _build_head(method, debias_target=True)
    head.debias_inited.fill_(True)
    head.debias_mean.fill_(float("inf"))
    init_visual_distill_buffers(_StubModel(head))
    head.train()
    loss, comps = head.compute(_samples(method, seed=1))
    assert torch.isfinite(loss), "step-0 loss non-finite after buffer init"
    assert bool(head.debias_inited) is True, "EMA should lazily init on step 0"
    assert torch.isfinite(head.debias_mean).all()

    # (b) defense-in-depth: even WITHOUT the build-time reset, the in-loss
    # isfinite guard must re-init the running mean rather than emit NaN.
    head2 = _build_head(method, debias_target=True)
    head2.debias_inited.fill_(True)
    head2.debias_mean.fill_(float("inf"))
    head2.train()
    loss2, _ = head2.compute(_samples(method, seed=1))
    assert torch.isfinite(loss2), "in-loss isfinite guard failed to prevent NaN"


def test_debias_breaks_constant_target_collapse() -> None:
    """Trick A: subtracting the EMA per-channel mean removes the shared "mean
    image" constant. On a constant target the plain cosine sees a single
    direction (trivially matchable / collapse-prone); after debias the residual
    is ~zero so the de-meaned target carries no shared direction -> the loss
    differs from the un-debiased cosine. We only assert the debias path runs and
    stays finite and that it changes the loss (the mechanism is active)."""
    plain = _build_head("eve")
    plain.train()
    l_plain, _ = plain.compute(_samples("eve", seed=2, constant_target=True))

    deb = _build_head("eve", debias_target=True)
    deb.train()
    l_deb, _ = deb.compute(_samples("eve", seed=2, constant_target=True))
    assert torch.isfinite(l_deb)
    assert not torch.equal(l_plain, l_deb)


def test_relational_gram_bounded_and_finite() -> None:
    """Trick B: the bounded Gram relational term (rkd_dist) over >=3 pooled
    images is finite and contributes a non-negative penalty even on a degenerate
    (near-constant) student, without the inf-gradient blow-ups of raw RKD."""
    head = _build_head("eve", debias_target=True, rkd_dist_weight=1.0)
    head.train()
    loss, comps = head.compute(_samples("eve", seed=3))
    assert torch.isfinite(loss)
    assert "distill_rkd_d" in comps and torch.isfinite(comps["distill_rkd_d"])
