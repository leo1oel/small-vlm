"""Flow-matching math for the text->image generation pathway (spec 2026-06-20).

Convention (minit2i / MM-JiT): t=1 is the CLEAN image, t=0 is pure noise.
  x_t   = x1 * t + noise * (1 - t)        # noise already scaled by noise_scale
  target velocity v = (x1 - x_t) / max(1-t, 0.05)   == x1 - noise  (away from t->1)
  loss  = MSE(v_pred, v_target) in patch/pixel space
The model predicts the CLEAN image (x-prediction); the loss is taken in v-space.
"""

import importlib.util
import pathlib

import torch

# Direct file-path import: gen_diffusion is pure-torch with no intra-package
# deps, so we bypass the (very slow ~90s) `vlm` package __init__ for fast TDD.
_GD = pathlib.Path(__file__).resolve().parents[1] / "src" / "vlm" / "models" / "gen_diffusion.py"
_spec = importlib.util.spec_from_file_location("gen_diffusion_under_test", _GD)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
add_noise = _m.add_noise
euler_step = _m.euler_step
flow_matching_loss = _m.flow_matching_loss
sample_timesteps = _m.sample_timesteps
to_velocity = _m.to_velocity
timestep_embedding = _m.timestep_embedding
GenTimestepEmbedder = _m.GenTimestepEmbedder


def test_add_noise_t1_is_clean():
    x1 = torch.randn(2, 4, 8)
    noise = torch.randn_like(x1)
    x_t, used = add_noise(x1, torch.ones(2), noise_scale=2.0, noise=noise)
    assert torch.allclose(x_t, x1, atol=1e-5)
    assert used is noise


def test_add_noise_t0_is_noise():
    x1 = torch.randn(2, 4, 8)
    noise = torch.randn_like(x1)
    x_t, _ = add_noise(x1, torch.zeros(2), noise_scale=2.0, noise=noise)
    assert torch.allclose(x_t, noise, atol=1e-5)


def test_add_noise_samples_scaled_noise_when_none():
    torch.manual_seed(0)
    x1 = torch.zeros(1, 50_000)
    _, noise = add_noise(x1, torch.zeros(1), noise_scale=2.0)
    assert noise.shape == x1.shape
    assert abs(noise.std().item() - 2.0) < 0.1  # std == noise_scale


def test_add_noise_broadcasts_per_sample_t():
    x1 = torch.randn(3, 6912)
    noise = torch.randn_like(x1)
    t = torch.tensor([0.0, 0.5, 1.0])
    x_t, _ = add_noise(x1, t, noise_scale=1.0, noise=noise)
    assert torch.allclose(x_t[0], noise[0], atol=1e-5)  # t=0
    assert torch.allclose(x_t[2], x1[2], atol=1e-5)  # t=1
    assert torch.allclose(x_t[1], 0.5 * x1[1] + 0.5 * noise[1], atol=1e-5)


def test_target_velocity_equals_x1_minus_noise():
    x1 = torch.randn(2, 16)
    noise = torch.randn_like(x1)
    t = torch.full((2,), 0.5)
    x_t, _ = add_noise(x1, t, noise_scale=1.0, noise=noise)
    v = to_velocity(x1, x_t, t)  # pred==x1 -> velocity == target velocity == x1-noise
    assert torch.allclose(v, x1 - noise, atol=1e-4)


def test_flow_matching_loss_zero_for_perfect_prediction():
    x1 = torch.randn(2, 16)
    noise = torch.randn_like(x1)
    t = torch.full((2,), 0.3)
    x_t, _ = add_noise(x1, t, noise_scale=2.0, noise=noise)
    loss = flow_matching_loss(x1, x_t, x1, t)  # pred_x0 == x1
    assert loss.item() < 1e-8


def test_flow_matching_loss_positive_for_wrong_prediction():
    x1 = torch.randn(2, 16)
    noise = torch.randn_like(x1)
    t = torch.full((2,), 0.3)
    x_t, _ = add_noise(x1, t, noise_scale=2.0, noise=noise)
    loss = flow_matching_loss(torch.zeros_like(x1), x_t, x1, t)
    assert loss.item() > 0.0


def test_flow_matching_loss_clamps_near_t1():
    # 1-t = 0.001 < 0.05 floor -> no blow-up
    x1 = torch.randn(2, 16)
    noise = torch.randn_like(x1)
    t = torch.full((2,), 0.999)
    x_t, _ = add_noise(x1, t, noise_scale=2.0, noise=noise)
    loss = flow_matching_loss(torch.zeros_like(x1), x_t, x1, t)
    assert torch.isfinite(loss).item()


def test_sample_timesteps_shape_and_range():
    torch.manual_seed(0)
    t = sample_timesteps(2000, mu=-0.8, sigma=0.8, device=torch.device("cpu"))
    assert t.shape == (2000,)
    assert (t > 0).all().item() and (t < 1).all().item()
    assert t.median().item() < 0.5  # mu<0 biases toward noisier (small t)


def test_euler_step():
    x = torch.randn(2, 8)
    v = torch.randn(2, 8)
    assert torch.allclose(euler_step(x, v, 0.1), x + v * 0.1)


def test_timestep_embedding_shape_and_bounds():
    t = torch.tensor([0.0, 0.3, 0.7, 1.0])
    emb = timestep_embedding(t, 256)
    assert emb.shape == (4, 256)
    assert torch.isfinite(emb).all().item()
    assert emb.abs().max().item() <= 1.0 + 1e-5  # cos/sin bounded


def test_timestep_embedding_distinct_for_distinct_t():
    emb = timestep_embedding(torch.tensor([0.1, 0.9]), 256)
    assert not torch.allclose(emb[0], emb[1])


def test_gen_timestep_embedder_shape_and_distinct():
    torch.manual_seed(0)
    embedder = GenTimestepEmbedder(64)
    out = embedder(torch.tensor([0.1, 0.5, 0.9]))
    assert out.shape == (3, 64)
    assert not torch.allclose(out[0], out[2])
