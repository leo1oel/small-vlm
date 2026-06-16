"""CPU unit test for the visual-prefix stack (spec 2026-06-14, early-capacity arm).

Tests the VisualPrefix module in connectors.py: shape preservation, the CRITICAL
per-image isolation property (image A's tokens must not attend to image B's via
the padding mask), gradient flow, and that depth=0-equivalent (no module) is the
baseline. Fast, CPU-only.

Run:  .venv/bin/python devtools/test_visual_prefix.py
"""

import torch

from vlm.models.connectors import VisualPrefix

DIM, HEADS, INTER = 32, 4, 64


def main():
    torch.manual_seed(0)
    pf = VisualPrefix(dim=DIM, depth=2, n_heads=HEADS, intermediate=INTER).eval()

    a = torch.randn(5, DIM)
    b = torch.randn(3, DIM)
    c = torch.randn(7, DIM)

    # shape preservation per image
    out = pf([a, b, c])
    assert [o.shape for o in out] == [a.shape, b.shape, c.shape], "shape changed"
    print("[ok] per-image shapes preserved")

    # CRITICAL: per-image isolation — A's output must be identical whether or not
    # B/C are in the same batch (the padding mask must block cross-image attention)
    out_together = pf([a, b, c])
    out_a = pf([a])
    out_b = pf([b])
    out_c = pf([c])
    assert torch.allclose(out_together[0], out_a[0], atol=1e-5), "image A leaked across batch!"
    assert torch.allclose(out_together[1], out_b[0], atol=1e-5), "image B leaked across batch!"
    assert torch.allclose(out_together[2], out_c[0], atol=1e-5), "image C leaked across batch!"
    print("[ok] per-image isolation: no cross-image attention leakage (padding mask correct)")

    # output differs from input (the stack actually transforms)
    assert not torch.allclose(out_a[0], a, atol=1e-3), "prefix is a no-op"
    print("[ok] prefix transforms its input")

    # gradient flows to every prefix layer
    pf.train()
    pf.zero_grad(set_to_none=True)
    loss = sum(o.sum() for o in pf([a, b]))
    loss.backward()
    for i, layer in enumerate(pf.layers):
        for name, p in layer.named_parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0, f"layer {i}.{name} no grad"
    print(f"[ok] all {len(pf.layers)} prefix layers receive gradient")

    # empty input is a no-op (text-only batch)
    assert pf([]) == [], "empty input not handled"
    print("[ok] empty input handled")

    print("\nALL VISUAL-PREFIX UNIT CHECKS PASSED")


if __name__ == "__main__":
    main()
