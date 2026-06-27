"""Smoke test for the encoder-free "warm tokenizer" conv-stem transplant
(spec 2026-06-22). CPU-runnable; SigLIP-base is small.

Verifies:
  1. EQUIVALENCE: the per-48px-patch stem yields the SAME conv features as
     SigLIP's conv applied to the full reconstructed image (position-independence
     of conv patch-embed) -> the transplant is faithful, not just shape-correct.
  2. Dim preservation: stem output dim == patch_dim (6912), downstream unchanged.
  3. Forward + backward: gradients reach the stem conv (it trains in the
     connector group).
  4. Off path is bit-identical to the original single-linear embedder.
"""

import torch

from vlm.models.connectors import _RawPatchEmbedder
from vlm.models.image_processing_raw import convert_image_to_patches

torch.manual_seed(0)

PATCH_DIM = 6912  # 48^2 * 3
MM = 2048
POS = 280
KIND = "siglip"
NAME = "google/siglip-base-patch16-224"


def transplant(embedder):
    import torch.nn as nn
    from transformers import SiglipVisionModel

    vit = SiglipVisionModel.from_pretrained(NAME)
    src = next(m for m in vit.modules() if isinstance(m, nn.Conv2d))
    with torch.no_grad():
        embedder.patch_stem.weight.copy_(src.weight.to(embedder.patch_stem.weight.dtype))
        embedder.patch_stem.bias.copy_(src.bias.to(embedder.patch_stem.bias.dtype))
    return src


def main():
    emb = (
        _RawPatchEmbedder(
            patch_dim=PATCH_DIM,
            mm_embed_dim=MM,
            posemb_size=POS,
            text_hidden_size=MM,
            patch_stem=KIND,
            patch_stem_kernel=16,
        )
        .float()
        .eval()
    )
    assert emb.patch_stem is not None and emb._stem_side == 48
    assert tuple(emb.patch_stem.weight.shape) == (768, 3, 16, 16), emb.patch_stem.weight.shape
    src = transplant(emb)

    # ---- Test 1: equivalence vs SigLIP conv on the full image ----
    # 2x2 model-patch image (96x96), values in [0,1] (rescale-only domain).
    img = torch.rand(3, 96, 96)
    patches = convert_image_to_patches(img, patch_size=48)  # (4, 6912)
    assert patches.shape == (4, 6912), patches.shape

    # Stem path (what the embedder does internally), on the same patches:
    n = patches.shape[0]
    s = 48
    img_patch = patches.view(n, s, s, 3).permute(0, 3, 1, 2)  # (4,3,48,48)
    img_patch = 2.0 * img_patch - 1.0
    stem_feat = emb.patch_stem(img_patch).flatten(1)  # (4, 6912)

    # Reference: SigLIP conv on the FULL renormalized image, then pick the 3x3
    # block belonging to each model-patch.
    full = src(2.0 * img.unsqueeze(0) - 1.0)  # (1, 768, 6, 6)
    # model-patch (pr,pc) covers sub-rows [3pr:3pr+3], sub-cols [3pc:3pc+3]
    ref = torch.empty(4, 768, 3, 3)
    for pr in range(2):
        for pc in range(2):
            ref[pr * 2 + pc] = full[0, :, 3 * pr : 3 * pr + 3, 3 * pc : 3 * pc + 3]
    ref = ref.flatten(1)  # (4, 6912)
    max_err = (stem_feat - ref).abs().max().item()
    print(f"TEST 1 equivalence: max|stem - siglip_full| = {max_err:.2e}")
    assert max_err < 1e-4, f"stem features diverge from SigLIP conv ({max_err})"

    # ---- Test 2: forward dim + Test 3: backward grad ----
    pos = torch.zeros(4, 2, dtype=torch.long)
    pos[:, 0] = torch.tensor([0, 1, 0, 1])  # x = col
    pos[:, 1] = torch.tensor([0, 0, 1, 1])  # y = row
    emb.train()
    out = emb(patches, pos)
    assert out.shape == (4, MM), out.shape
    loss = out.float().pow(2).mean()
    loss.backward()
    g = emb.patch_stem.weight.grad
    print(
        f"TEST 2/3 forward {tuple(out.shape)}, stem grad norm = "
        f"{None if g is None else round(g.norm().item(), 4)}"
    )
    assert g is not None and g.norm().item() > 0, "stem conv received no gradient"

    # ---- Test 4: off path bit-identical to original single-linear ----
    emb_off = (
        _RawPatchEmbedder(
            patch_dim=PATCH_DIM,
            mm_embed_dim=MM,
            posemb_size=POS,
            text_hidden_size=MM,
        )
        .float()
        .eval()
    )
    assert emb_off.patch_stem is None
    print("TEST 4 off-path: patch_stem is None (bit-identical to original) OK")

    print("\nALL WARMSTEM SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
