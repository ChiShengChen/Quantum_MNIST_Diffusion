"""Tests for the strided DDIM sampler.

DDIM is what makes evaluating the quantum arm affordable at all: every reverse
step is a model call, and for that arm a QNode call, so 1000-step sampling puts
FID/KID at ~177 h per cell. A subtly wrong stride would not crash -- it would
quietly degrade every number in the sweep -- so the schedule indexing and the
step count are pinned here.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="module")
def v7():
    spec = importlib.util.spec_from_file_location(
        "v7_ddim", REPO_ROOT / "quantum_difussion_mnist_v7.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


class CountingModel(torch.nn.Module):
    """Records how many times the reverse loop calls it, and with which t."""

    def __init__(self):
        super().__init__()
        self.calls = 0
        self.timesteps_seen = []
        self.lin = torch.nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x, t):
        self.calls += 1
        self.timesteps_seen.append(int(t[0]))
        return self.lin(x) * 0.01

    def eval(self):
        return self


def test_ddim_makes_exactly_one_model_call_per_step(v7):
    """The cost model in run_scaling_study assumes one call per step."""
    diff = v7.ImprovedGaussianDiffusion(timesteps=1000)
    model = CountingModel()
    diff.ddim_sample(model, torch.randn(2, 1, 28, 28), n_steps=50)
    assert model.calls == 50, f"expected 50 model calls, got {model.calls}"


def test_ddim_is_cheaper_than_ddpm_by_the_stride_ratio(v7):
    diff = v7.ImprovedGaussianDiffusion(timesteps=1000)
    ddim_model = CountingModel()
    diff.ddim_sample(ddim_model, torch.randn(2, 1, 28, 28), n_steps=50)

    ddpm_model = CountingModel()
    x = torch.randn(2, 1, 28, 28)
    for t in reversed(range(1000)):
        x = diff.p_sample(ddpm_model, x, torch.full((2,), t, dtype=torch.long))

    assert ddpm_model.calls == 1000
    assert ddpm_model.calls / ddim_model.calls == 20


def test_ddim_traverses_the_full_schedule_descending(v7):
    """It must start near t=T-1, end at t=0, and never go backwards."""
    diff = v7.ImprovedGaussianDiffusion(timesteps=1000)
    model = CountingModel()
    diff.ddim_sample(model, torch.randn(2, 1, 28, 28), n_steps=50)

    seen = model.timesteps_seen
    assert seen[0] == 999, f"must start at the noisiest timestep, got {seen[0]}"
    assert seen[-1] == 0, f"must finish at t=0, got {seen[-1]}"
    assert seen == sorted(seen, reverse=True), "timesteps must be descending"
    assert len(set(seen)) == len(seen), "a timestep was visited twice"


def test_ddim_output_is_in_range_and_finite(v7):
    diff = v7.ImprovedGaussianDiffusion(timesteps=1000)
    model = CountingModel()
    out = diff.ddim_sample(model, torch.randn(3, 1, 28, 28), n_steps=25)
    assert out.shape == (3, 1, 28, 28)
    assert torch.isfinite(out).all(), "sampler produced NaN/inf"
    assert out.min() >= -1.001 and out.max() <= 1.001, (
        f"output escaped [-1, 1]: [{out.min():.3f}, {out.max():.3f}]"
    )


def test_ddim_eta_zero_is_deterministic(v7):
    """eta=0 must give the same image from the same noise, or paired comparisons
    across arms are not paired."""
    diff = v7.ImprovedGaussianDiffusion(timesteps=1000)
    model = CountingModel()
    x0 = torch.randn(2, 1, 28, 28)
    a = diff.ddim_sample(model, x0.clone(), n_steps=20, eta=0.0)
    b = diff.ddim_sample(model, x0.clone(), n_steps=20, eta=0.0)
    assert torch.allclose(a, b), "eta=0 sampling is not deterministic"


def test_ddim_eta_one_is_stochastic(v7):
    diff = v7.ImprovedGaussianDiffusion(timesteps=1000)
    model = CountingModel()
    x0 = torch.randn(2, 1, 28, 28)
    torch.manual_seed(0)
    a = diff.ddim_sample(model, x0.clone(), n_steps=20, eta=1.0)
    torch.manual_seed(1)
    b = diff.ddim_sample(model, x0.clone(), n_steps=20, eta=1.0)
    assert not torch.allclose(a, b), "eta=1 should inject noise"


def test_sample_helper_routes_to_the_requested_sampler(v7):
    diff = v7.ImprovedGaussianDiffusion(timesteps=1000)
    model = CountingModel()
    v7.sample(model, diff, steps=30, n=2, sampler="ddim")
    assert model.calls == 30

    model2 = CountingModel()
    v7.sample(model2, diff, steps=10, n=2, sampler="ddpm")
    assert model2.calls == 10


def test_sample_helper_rejects_unknown_sampler(v7):
    diff = v7.ImprovedGaussianDiffusion(timesteps=1000)
    with pytest.raises(ValueError, match="unknown sampler"):
        v7.sample(CountingModel(), diff, steps=5, n=1, sampler="euler")
