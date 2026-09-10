"""Tests for the tensor-based evaluation pipeline.

These use a tiny stub feature extractor rather than InceptionV3, so the suite
stays fast and does not download 100MB of weights. What is being tested is the
plumbing -- conversion, batching, and that the metrics respond in the right
direction -- not Inception itself.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import evaluate  # noqa: E402


class TinyFeatures(nn.Module):
    """Stand-in for InceptionV3: deterministic, cheap, 16-dim output.

    torchmetrics requires a module that maps a uint8 image batch to [N, F].
    """

    num_features = 16

    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

    def forward(self, x):
        x = x.float() / 255.0
        # [N, 3, H, W] -> [N, 3, 4, 4] -> [N, 48] -> first 16 dims via mean over channels
        return self.pool(x).flatten(1)[:, : self.num_features]


# --------------------------------------------------------------------------- #
# to_uint8
# --------------------------------------------------------------------------- #

def test_to_uint8_maps_range_endpoints():
    x = torch.tensor([-1.0, 0.0, 1.0]).view(1, 1, 1, 3)
    out = evaluate.to_uint8(x)
    assert out.dtype == torch.uint8
    assert out[0, 0, 0].tolist() == [0, 128, 255]


def test_to_uint8_expands_grayscale_to_three_channels():
    out = evaluate.to_uint8(torch.zeros(4, 1, 28, 28))
    assert out.shape == (4, 3, 28, 28)
    # all three channels must be identical copies
    assert torch.equal(out[:, 0], out[:, 1]) and torch.equal(out[:, 1], out[:, 2])


def test_to_uint8_passes_rgb_through():
    out = evaluate.to_uint8(torch.zeros(2, 3, 28, 28))
    assert out.shape == (2, 3, 28, 28)


def test_to_uint8_clamps_out_of_range_values():
    """A diffusion sampler can emit values outside [-1, 1]; they must not wrap."""
    x = torch.tensor([-5.0, 5.0]).view(1, 1, 1, 2)
    assert evaluate.to_uint8(x)[0, 0, 0].tolist() == [0, 255]


def test_to_uint8_rejects_wrong_rank():
    with pytest.raises(ValueError, match=r"\[N, C, H, W\]"):
        evaluate.to_uint8(torch.zeros(3, 28, 28))


def test_to_uint8_rejects_odd_channel_count():
    with pytest.raises(ValueError, match="1 or 3 channels"):
        evaluate.to_uint8(torch.zeros(2, 2, 28, 28))


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def _metrics(gen, real):
    return evaluate.compute_metrics(
        gen, real, device=torch.device("cpu"),
        kid_subset_size=8, feature=TinyFeatures(),
    )


def test_identical_distributions_score_near_zero():
    torch.manual_seed(0)
    images = evaluate.to_uint8(torch.randn(64, 1, 28, 28).clamp(-1, 1))
    result = _metrics(images, images.clone())
    assert result["fid"] == pytest.approx(0.0, abs=1e-3)
    assert result["kid_mean"] == pytest.approx(0.0, abs=1e-3)


def test_shifted_distribution_scores_worse_than_matched_one():
    """The metric must move in the right direction under a real distribution shift."""
    torch.manual_seed(0)
    real = evaluate.to_uint8(torch.randn(64, 1, 28, 28).clamp(-1, 1) * 0.2)
    matched = evaluate.to_uint8(torch.randn(64, 1, 28, 28).clamp(-1, 1) * 0.2)
    shifted = evaluate.to_uint8((torch.randn(64, 1, 28, 28) * 0.2 + 0.8).clamp(-1, 1))

    assert _metrics(shifted, real)["fid"] > _metrics(matched, real)["fid"]
    assert _metrics(shifted, real)["kid_mean"] > _metrics(matched, real)["kid_mean"]


def test_metrics_report_sample_counts():
    torch.manual_seed(0)
    gen = evaluate.to_uint8(torch.randn(16, 1, 28, 28).clamp(-1, 1))
    real = evaluate.to_uint8(torch.randn(32, 1, 28, 28).clamp(-1, 1))
    result = _metrics(gen, real)
    assert result["n_generated"] == 16
    assert result["n_real"] == 32


def test_batching_does_not_change_the_result():
    """_feed() splits into batches; the metric must be batch-size invariant."""
    torch.manual_seed(0)
    gen = evaluate.to_uint8(torch.randn(48, 1, 28, 28).clamp(-1, 1))
    real = evaluate.to_uint8(torch.randn(48, 1, 28, 28).clamp(-1, 1) * 0.5)

    from torchmetrics.image.fid import FrechetInceptionDistance

    scores = []
    for batch_size in (7, 48):
        metric = FrechetInceptionDistance(feature=TinyFeatures(), normalize=False)
        evaluate._feed(metric, real, torch.device("cpu"), real=True, batch_size=batch_size)
        evaluate._feed(metric, gen, torch.device("cpu"), real=False, batch_size=batch_size)
        scores.append(float(metric.compute()))

    assert scores[0] == pytest.approx(scores[1], rel=1e-4)


# --------------------------------------------------------------------------- #
# CLI smoke tests
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("script", ["evaluate.py", "generate_samples.py"])
def test_cli_help(script):
    """--help must work without downloading datasets or model weights."""
    proc = subprocess.run([sys.executable, str(REPO_ROOT / script), "--help"],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "usage:" in proc.stdout


def test_generate_samples_rejects_mismatched_checkpoint(tmp_path):
    """Loading a classical checkpoint as --use-quantum must fail loudly."""
    import generate_samples

    module, _ = generate_samples.load_arch("v7_mnist")
    classical = module.ImprovedUNet(use_quantum=False)
    ckpt = tmp_path / "classical.pth"
    torch.save(classical.state_dict(), ckpt)

    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "generate_samples.py"),
         "--arch", "v7_mnist", "--checkpoint", str(ckpt),
         "--out", str(tmp_path / "out.pt"), "--n", "1", "--steps", "1",
         "--use-quantum", "--device", "cpu"],
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode != 0
    assert "checkpoint does not match" in proc.stderr
    assert not (tmp_path / "out.pt").exists()


def test_generate_samples_end_to_end(tmp_path):
    """A tiny end-to-end run must write a correctly shaped, in-range tensor."""
    import generate_samples

    module, _ = generate_samples.load_arch("v7_mnist")
    ckpt = tmp_path / "model.pth"
    torch.save(module.ImprovedUNet(use_quantum=False).state_dict(), ckpt)
    out = tmp_path / "samples.pt"

    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "generate_samples.py"),
         "--arch", "v7_mnist", "--checkpoint", str(ckpt), "--out", str(out),
         "--n", "3", "--steps", "2", "--batch-size", "2", "--device", "cpu"],
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr

    samples = torch.load(out)
    assert samples.shape == (3, 1, 28, 28)
    assert evaluate.to_uint8(samples).shape == (3, 3, 28, 28)
