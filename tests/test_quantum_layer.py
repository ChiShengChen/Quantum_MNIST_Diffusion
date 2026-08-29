"""Regression tests for the quantum layer's trainability.

These pin down two bugs that made the "quantum" configuration untrainable:

1.  The variational circuit was built exclusively from gates that are diagonal in
    the computational basis (``RZ``) and gates that permute computational basis
    states (``CNOT``).  Neither can change ``|<b|psi>|^2``, so every
    ``expval(PauliZ)`` was *exactly* independent of ``weights`` and the gradient
    with respect to all 48 circuit parameters was identically zero.

2.  ``QuantumLayer.forward`` called ``.detach()`` on its input, which removed the
    quantum branch from the autograd graph on the input side.  ``input_proj``
    (v7) and the encoder (v8) therefore never received a gradient.

Run with:  pytest tests/ -v
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

MODULES = {
    "v7_mnist": REPO_ROOT / "quantum_difussion_mnist_v7.py",
    "v7_pathmnist": REPO_ROOT / "quantum_difussion_pathmnist_v7.py",
    "v8_mnist": REPO_ROOT / "full_unet" / "quantum_diffusion_mnist_v8.py",
}

# Keep the circuits small so the suite stays fast; the bugs are independent of width.
N_QUBITS = 4
N_LAYERS = 2

# Backprop through the state-vector simulator leaves float32 round-off in the
# weight gradient even when the analytic gradient is exactly zero.  Measured on
# the pre-fix circuit that floor is ~9e-8 (max |grad|); after the fix the same
# setup gives ~1.5.  Anything below this threshold is noise, not learning signal
# -- asserting merely `> 0` would let the original bug pass.
GRAD_NOISE_FLOOR = 1e-4


def _load(name):
    """Import one of the training scripts by path (they are not a package)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, MODULES[name])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:  # e.g. medmnist not installed
        del sys.modules[name]
        pytest.skip(f"cannot import {name}: {exc}")
    return module


@pytest.fixture(params=sorted(MODULES))
def module(request):
    return _load(request.param)


def _make_layer(module):
    """Build a small QuantumLayer; v8 takes an extra embed_dim argument."""
    try:
        return module.QuantumLayer(n_qubits=N_QUBITS, n_layers=N_LAYERS)
    except TypeError:
        return module.QuantumLayer(n_qubits=N_QUBITS, n_layers=N_LAYERS, embed_dim=128)


def _feature_dim(module):
    """Input width the layer expects (128 in every current variant)."""
    layer = _make_layer(module)
    return getattr(layer, "input_proj", None).in_features if hasattr(layer, "input_proj") else 128


# --------------------------------------------------------------------------- #
# Bug 1: the circuit's parameters must actually affect the circuit's output
# --------------------------------------------------------------------------- #

def test_circuit_output_depends_on_weights(module):
    """Two different weight tensors must produce different measurements.

    Before the fix this assertion failed with a difference of ~1e-16: RZ + CNOT
    only ever multiplies amplitudes by phases and permutes basis labels, so the
    PauliZ expectations were a fixed function of the inputs alone.
    """
    torch.manual_seed(0)
    layer = _make_layer(module)
    x = torch.randn(3, _feature_dim(module))

    with torch.no_grad():
        torch.nn.init.uniform_(layer.qlayer.weights, -torch.pi, torch.pi)
        first = layer(x).clone()
        torch.nn.init.uniform_(layer.qlayer.weights, -torch.pi, torch.pi)
        second = layer(x).clone()

    assert not torch.allclose(first, second, atol=1e-6), (
        "QuantumLayer output is invariant to its own weights - the circuit is "
        "built only from basis-diagonal and basis-permutation gates."
    )


def test_circuit_weights_receive_gradient(module):
    """d(loss)/d(circuit weights) must be non-zero."""
    torch.manual_seed(0)
    layer = _make_layer(module)
    x = torch.randn(3, _feature_dim(module))

    layer(x).sum().backward()

    grad = layer.qlayer.weights.grad
    assert grad is not None, "circuit weights received no gradient at all"
    assert torch.isfinite(grad).all(), "circuit weight gradient contains NaN/inf"
    assert grad.abs().max() > GRAD_NOISE_FLOOR, (
        f"circuit weight gradient is {grad.abs().max():.2e}, at or below the "
        f"simulator's float32 noise floor ({GRAD_NOISE_FLOOR:.0e}) - the circuit "
        "parameters are not actually being trained."
    )


# --------------------------------------------------------------------------- #
# Bug 2: the gradient must reach the input side of the layer
# --------------------------------------------------------------------------- #

def test_gradient_flows_through_to_layer_input(module):
    """The tensor fed to QuantumLayer must stay attached to the autograd graph."""
    torch.manual_seed(0)
    layer = _make_layer(module)
    x = torch.randn(3, _feature_dim(module), requires_grad=True)

    layer(x).sum().backward()

    assert x.grad is not None, "input to QuantumLayer was detached from the graph"
    assert x.grad.abs().max() > 0, "input gradient is identically zero"


def test_input_projection_receives_gradient(module):
    """v7's input_proj was frozen at initialization by the .detach() call."""
    layer = _make_layer(module)
    if not hasattr(layer, "input_proj"):
        pytest.skip("this variant has no input_proj")

    torch.manual_seed(0)
    layer(torch.randn(3, layer.input_proj.in_features)).sum().backward()

    grad = layer.input_proj.weight.grad
    assert grad is not None, "input_proj received no gradient (detached input)"
    assert grad.abs().max() > 0, "input_proj gradient is identically zero"


# --------------------------------------------------------------------------- #
# Batching: v7 used a per-sample Python loop; the circuit must accept [B, n]
# --------------------------------------------------------------------------- #

def test_batched_forward_matches_per_sample(module):
    """A batched call must equal looping over the batch one sample at a time."""
    torch.manual_seed(0)
    layer = _make_layer(module).eval()
    x = torch.randn(4, _feature_dim(module))

    with torch.no_grad():
        batched = layer(x)
        per_sample = torch.cat([layer(x[i : i + 1]) for i in range(x.shape[0])])

    assert batched.shape == (4, _feature_dim(module))
    assert torch.allclose(batched, per_sample, atol=1e-5)


# --------------------------------------------------------------------------- #
# End to end: every quantum parameter must train inside the full U-Net
# --------------------------------------------------------------------------- #

def test_unet_backward_reaches_every_quantum_parameter(module):
    """One backward pass through the U-Net must give every q_attn parameter a gradient."""
    torch.manual_seed(0)
    model = module.ImprovedUNet(use_quantum=True)
    # Shrink the circuit so the test is fast, keeping the wiring identical.
    model.q_attn = _make_layer(module)

    channels = model.enc1.in_channels if hasattr(model, "enc1") else model.init_conv.in_channels
    x = torch.randn(2, channels, 28, 28)
    t = torch.randint(0, 1000, (2,))

    model(x, t).sum().backward()

    for name, param in model.q_attn.named_parameters():
        assert param.grad is not None, f"q_attn.{name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"q_attn.{name} gradient has NaN/inf"
        # The circuit parameters get the strict threshold: their pre-fix gradient
        # was non-zero only through simulator round-off.
        floor = GRAD_NOISE_FLOOR if name.endswith("weights") else 0.0
        assert param.grad.abs().max() > floor, (
            f"q_attn.{name} gradient is {param.grad.abs().max():.2e}, which is not "
            "a usable learning signal"
        )
