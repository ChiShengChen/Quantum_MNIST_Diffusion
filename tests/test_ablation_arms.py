"""Tests for the #3 ablation ladder and the #5 data-scaling machinery.

The point of the ladder is that `se` and `quantum` differ *only* by the QNode in
place of the ReLU, so a measured difference is attributable to the circuit rather
than to the existence of channel gating. These tests pin that down numerically,
because a silent drift in either arm's width would quietly re-introduce the
confound the ladder exists to remove.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import bottlenecks  # noqa: E402


def _v7():
    spec = importlib.util.spec_from_file_location(
        "v7_arms", REPO_ROOT / "quantum_difussion_mnist_v7.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def v7():
    return _v7()


# --------------------------------------------------------------------------- #
# The ladder itself
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("arm", bottlenecks.ARMS)
def test_every_arm_builds_and_runs(v7, arm):
    torch.manual_seed(0)
    net = v7.ImprovedUNet(arm=arm, n_hidden=4)
    out = net(torch.randn(2, 1, 28, 28), torch.randint(0, 1000, (2,)))
    assert out.shape == (2, 1, 28, 28)


def test_plain_arm_has_no_bottleneck(v7):
    net = v7.ImprovedUNet(arm="plain")
    assert not hasattr(net, "q_attn"), "the plain arm must have no gating module"
    assert net.bottleneck_parameter_counts() == (0, 0)


def test_se_and_quantum_are_parameter_matched(v7):
    """They must differ by exactly the circuit's own parameters, nothing else.

    If this drifts, `se` stops being a parameter-matched control and the
    comparison silently regains the confound described in issue #3.
    """
    se = v7.ImprovedUNet(arm="se", n_hidden=16)
    qt = v7.ImprovedUNet(arm="quantum", n_hidden=16)

    se_params, _ = se.bottleneck_parameter_counts()
    qt_params, _ = qt.bottleneck_parameter_counts()
    circuit_params = qt.q_attn.qlayer.weights.numel()

    assert qt_params - se_params == circuit_params, (
        f"se={se_params}, quantum={qt_params}, circuit={circuit_params}: the two "
        "arms differ by something other than the circuit parameters"
    )


def test_se_frozen_keeps_down_projection_frozen_through_training(v7):
    """The frozen arm's down-projection must not move under an optimizer step."""
    torch.manual_seed(0)
    net = v7.ImprovedUNet(arm="se_frozen", n_hidden=4)
    before = net.q_attn.down.weight.detach().clone()

    opt = torch.optim.Adam(net.parameters(), lr=1e-1)
    net(torch.randn(2, 1, 28, 28), torch.randint(0, 1000, (2,))).sum().backward()
    opt.step()

    assert torch.equal(net.q_attn.down.weight, before), "frozen down-projection moved"
    assert net.q_attn.down.weight.requires_grad is False
    # ...while the up-projection must still learn, or the arm is inert.
    assert net.q_attn.up.weight.grad is not None
    assert net.q_attn.up.weight.grad.abs().max() > 0


@pytest.mark.parametrize("arm", ("se", "se_frozen", "quantum"))
def test_gating_arms_receive_gradient(v7, arm):
    torch.manual_seed(0)
    net = v7.ImprovedUNet(arm=arm, n_hidden=4)
    net(torch.randn(2, 1, 28, 28), torch.randint(0, 1000, (2,))).sum().backward()
    trainable = [(n, p) for n, p in net.q_attn.named_parameters() if p.requires_grad]
    assert trainable
    for name, p in trainable:
        assert p.grad is not None, f"{arm}: q_attn.{name} got no gradient"
        assert p.grad.abs().max() > 0, f"{arm}: q_attn.{name} gradient is zero"


def test_unknown_arm_is_rejected(v7):
    with pytest.raises(ValueError, match="unknown arm"):
        v7.ImprovedUNet(arm="quantuum")


def test_use_quantum_remains_a_working_alias(v7):
    """Existing callers pass use_quantum=; it must keep selecting the same arms."""
    assert v7.ImprovedUNet(use_quantum=True).arm == "quantum"
    assert v7.ImprovedUNet(use_quantum=False).arm == "plain"
    assert v7.ImprovedUNet(use_quantum=True).use_quantum is True
    # An explicit arm wins over the legacy flag.
    assert v7.ImprovedUNet(use_quantum=False, arm="se").arm == "se"


# --------------------------------------------------------------------------- #
# Data scaling (#5)
# --------------------------------------------------------------------------- #

class _FakeMNIST:
    """Stand-in for the dataset, so these tests need no download."""

    def __init__(self, n_per_class=50, n_classes=3):
        self.data = torch.arange(n_per_class * n_classes).view(-1, 1, 1).repeat(1, 28, 28)
        self.targets = torch.arange(n_per_class * n_classes) % n_classes


def test_subsample_returns_requested_size(v7):
    ds, n = v7.subsample(_FakeMNIST(), digit_label=1, n_train=10, seed=0)
    assert n == 10 and len(ds.data) == 10
    assert (ds.targets == 1).all(), "subsample must not leak other classes"


def test_subsample_is_seed_dependent(v7):
    a, _ = v7.subsample(_FakeMNIST(), 1, 5, seed=0)
    b, _ = v7.subsample(_FakeMNIST(), 1, 5, seed=1)
    fa = sorted(a.data[:, 0, 0].tolist())
    fb = sorted(b.data[:, 0, 0].tolist())
    assert fa != fb, (
        "the data subset must move with the seed -- at small N which images you "
        "draw dominates every other source of variance (issue #5)"
    )


def test_subsample_is_reproducible_for_one_seed(v7):
    a, _ = v7.subsample(_FakeMNIST(), 1, 5, seed=7)
    b, _ = v7.subsample(_FakeMNIST(), 1, 5, seed=7)
    assert torch.equal(a.data, b.data)


def test_subsample_caps_at_available_data(v7):
    ds, n = v7.subsample(_FakeMNIST(n_per_class=20), 1, n_train=10**6, seed=0)
    assert n == 20 and len(ds.data) == 20


def test_subsample_none_keeps_whole_class(v7):
    ds, n = v7.subsample(_FakeMNIST(n_per_class=20), 1, n_train=None, seed=0)
    assert n == 20


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def test_cli_help_lists_every_arm():
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "quantum_difussion_mnist_v7.py"), "--help"],
        capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    for arm in bottlenecks.ARMS:
        assert arm in proc.stdout, f"--help does not mention the {arm} arm"


def test_cli_rejects_unknown_arm():
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "quantum_difussion_mnist_v7.py"),
         "--arm", "nonsense"],
        capture_output=True, text=True, timeout=300)
    assert proc.returncode != 0
    assert "invalid choice" in proc.stderr


def test_build_bottleneck_rejects_unknown_arm():
    with pytest.raises(ValueError, match="unknown arm"):
        bottlenecks.build_bottleneck("nope", None)


# --------------------------------------------------------------------------- #
# Simulator backends must be interchangeable
# --------------------------------------------------------------------------- #

def test_lightning_adjoint_matches_default_backprop(v7):
    """The fast backend must give the same gradients, or the speedup is a trap.

    lightning.qubit + adjoint is ~3.5x faster at n_qubits=16 and is what makes
    the #5 grid affordable, but only if it is numerically equivalent.
    """
    pytest.importorskip("pennylane_lightning")
    import pennylane as qml
    try:
        qml.device("lightning.qubit", wires=2)
    except Exception as exc:
        pytest.skip(f"lightning.qubit unavailable: {exc}")

    torch.manual_seed(0)
    ref = v7.QuantumLayer(n_qubits=4, n_layers=2)
    fast = v7.QuantumLayer(n_qubits=4, n_layers=2,
                           device_name="lightning.qubit", diff_method="adjoint")
    fast.load_state_dict(ref.state_dict())

    x = torch.randn(3, 128)
    a, b = ref(x.clone().requires_grad_(True)), fast(x.clone().requires_grad_(True))
    assert torch.allclose(a, b, atol=1e-5), "forward pass differs between backends"

    ref(x.clone()).sum().backward()
    fast(x.clone()).sum().backward()
    ga, gb = ref.qlayer.weights.grad, fast.qlayer.weights.grad
    assert torch.allclose(ga, gb, atol=1e-4), (
        f"gradients differ between backends: max |diff| = {(ga - gb).abs().max():.2e}"
    )


def test_ema_model_is_constructible_for_any_backend(v7):
    """copy.deepcopy fails on lightning's C++ statevector; the training loop must
    not rely on it."""
    import copy as _copy
    net = v7.ImprovedUNet(arm="quantum", n_hidden=4,
                          device_name="lightning.qubit", diff_method="adjoint")
    with pytest.raises(TypeError, match="pickle"):
        _copy.deepcopy(net)          # documents *why* the loop builds + loads instead
    clone = v7.ImprovedUNet(arm="quantum", n_hidden=4,
                            device_name="lightning.qubit", diff_method="adjoint")
    clone.load_state_dict(net.state_dict())   # the path train_pipeline actually uses
    for (n1, p1), (n2, p2) in zip(net.named_parameters(), clone.named_parameters()):
        assert n1 == n2 and torch.equal(p1, p2)


# --------------------------------------------------------------------------- #
# The ladder across all three model scripts (#3 asks for v7 *and* v8)
# --------------------------------------------------------------------------- #

def _load_path(tag, relpath):
    spec = importlib.util.spec_from_file_location(tag, REPO_ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[tag] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        pytest.skip(f"cannot import {relpath}: {type(exc).__name__}: {exc}")
    return module


# (tag, path, extra ctor kwargs, input channels, arms that should be available)
SCRIPTS = [
    ("v7_mnist", "quantum_difussion_mnist_v7.py", {}, 1,
     ("plain", "se", "se_frozen", "quantum")),
    ("pathmnist", "quantum_difussion_pathmnist_v7.py", {"channels": 3}, 3,
     ("plain", "se", "se_frozen", "quantum")),
    # v8's circuit pools its own input, so there is no down-projection to freeze.
    ("v8_mnist", "full_unet/quantum_diffusion_mnist_v8.py", {}, 1,
     ("plain", "se", "quantum")),
]


@pytest.fixture(params=SCRIPTS, ids=[s[0] for s in SCRIPTS])
def script(request):
    tag, path, kwargs, channels, arms = request.param
    return _load_path(tag, path), kwargs, channels, arms


def test_se_is_parameter_matched_to_quantum_in_every_script(script):
    """quantum - se must equal the circuit's own parameter count, everywhere.

    This is the invariant the whole ladder rests on. v8 needs a *pooling*
    down-step to satisfy it, because its QuantumLayer has no input_proj -- a
    Linear there would silently give the classical arm 2064 extra parameters.
    """
    mod, kwargs, _, _ = script
    se = mod.ImprovedUNet(arm="se", n_hidden=16, **kwargs)
    qt = mod.ImprovedUNet(arm="quantum", n_hidden=16, **kwargs)
    se_n, _ = se.bottleneck_parameter_counts()
    qt_n, _ = qt.bottleneck_parameter_counts()
    circuit = qt.q_attn.qlayer.weights.numel()
    assert qt_n - se_n == circuit, (
        f"se={se_n}, quantum={qt_n}, circuit={circuit}: the arms differ by "
        "something other than the circuit parameters"
    )


def test_every_available_arm_runs_in_every_script(script):
    mod, kwargs, channels, arms = script
    for arm in arms:
        torch.manual_seed(0)
        net = mod.ImprovedUNet(arm=arm, n_hidden=4, **kwargs)
        x = torch.randn(2, channels, 28, 28)
        out = net(x, torch.randint(0, 1000, (2,)))
        assert out.shape == (2, channels, 28, 28), f"{arm}: bad output shape"
        out.sum().backward()


def test_v8_rejects_se_frozen(script):
    """se_frozen must fail loudly on v8 rather than silently duplicating se."""
    mod, kwargs, _, arms = script
    if "se_frozen" in arms:
        pytest.skip("this script has a trainable input_proj, so se_frozen applies")
    with pytest.raises(ValueError, match="freeze_down is meaningless"):
        mod.ImprovedUNet(arm="se_frozen", n_hidden=4, **kwargs)


def test_use_quantum_alias_holds_in_every_script(script):
    mod, kwargs, _, _ = script
    assert mod.ImprovedUNet(use_quantum=True, n_hidden=4, **kwargs).arm == "quantum"
    assert mod.ImprovedUNet(use_quantum=False, **kwargs).arm == "plain"


# --------------------------------------------------------------------------- #
# SEBottleneck's two down-projection modes
# --------------------------------------------------------------------------- #

def test_pool_down_mode_has_no_down_parameters():
    """The pooling reduction must contribute nothing, matching v8's circuit."""
    se = bottlenecks.SEBottleneck(channels=128, n_hidden=16, down="pool")
    assert se.down is None
    names = [n for n, _ in se.named_parameters()]
    assert all(n.startswith("up.") for n in names), f"unexpected parameters: {names}"
    out = se(torch.randn(3, 128))
    assert out.shape == (3, 128)


def test_linear_and_pool_modes_differ_by_the_down_projection():
    lin = bottlenecks.SEBottleneck(channels=128, n_hidden=16, down="linear")
    pool = bottlenecks.SEBottleneck(channels=128, n_hidden=16, down="pool")
    lin_n, _ = bottlenecks.count_parameters(lin)
    pool_n, _ = bottlenecks.count_parameters(pool)
    assert lin_n - pool_n == 128 * 16 + 16


def test_pool_mode_rejects_freeze_down():
    with pytest.raises(ValueError, match="freeze_down is meaningless"):
        bottlenecks.SEBottleneck(down="pool", freeze_down=True)


def test_unknown_down_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown down"):
        bottlenecks.SEBottleneck(down="conv")


# --------------------------------------------------------------------------- #
# run_scaling_study: the budget caveat must be enforced, not just documented
# --------------------------------------------------------------------------- #

def test_runner_only_emits_step_budget_where_it_exists():
    import run_scaling_study as rss
    assert rss.supports_step_budget("quantum_difussion_mnist_v7.py")
    assert not rss.supports_step_budget("full_unet/quantum_diffusion_mnist_v8.py")
    assert not rss.supports_step_budget("quantum_difussion_pathmnist_v7.py")

    v7 = rss.command("quantum_difussion_mnist_v7.py", 3, 100, "se", 0, 2000, "runs", 16)
    v8 = rss.command("full_unet/quantum_diffusion_mnist_v8.py", 3, 100, "se", 0, 2000, "runs", 16)
    assert "--max-steps" in v7, "v7 must get the step budget"
    assert "--max-steps" not in v8, (
        "v8 has no --max-steps flag; emitting it would make every cell fail"
    )
    for cmd in (v7, v8):
        assert "--n-train" in cmd and "100" in cmd


def test_runner_warns_when_sweeping_n_without_a_step_budget(tmp_path):
    """Sweeping N on an epoch-based script conflates data with compute."""
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "run_scaling_study.py"),
         "--script", "full_unet/quantum_diffusion_mnist_v8.py",
         "--digits", "3", "--dry-run"],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert "no --max-steps" in proc.stderr
    assert "se_frozen" in proc.stderr, "v8 must drop the inapplicable arm"


@pytest.mark.parametrize("script", [
    "quantum_difussion_mnist_v7.py",
    "quantum_difussion_pathmnist_v7.py",
    "full_unet/quantum_diffusion_mnist_v8.py",
])
def test_every_training_script_has_a_working_cli(script):
    """The README documented --use_quantum flags that never existed; all three
    scripts now take real arguments."""
    proc = subprocess.run([sys.executable, str(REPO_ROOT / script), "--help"],
                          capture_output=True, text=True, timeout=300)
    if proc.returncode != 0 and "medmnist" in proc.stderr:
        pytest.skip("medmnist not installed")
    assert proc.returncode == 0, proc.stderr
    assert "--arm" in proc.stdout and "--n-train" in proc.stdout


def test_v8_cli_does_not_offer_se_frozen():
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "full_unet/quantum_diffusion_mnist_v8.py"),
         "--arm", "se_frozen"], capture_output=True, text=True, timeout=300)
    assert proc.returncode != 0
    assert "invalid choice" in proc.stderr


# --------------------------------------------------------------------------- #
# sweep_utils: one definition of the subsetting semantics
# --------------------------------------------------------------------------- #

def test_seeded_subset_indices_is_seed_dependent_and_reproducible():
    from sweep_utils import seeded_subset_indices
    targets = torch.arange(300) % 3
    a, na = seeded_subset_indices(targets, 1, 10, seed=0)
    b, _ = seeded_subset_indices(targets, 1, 10, seed=1)
    c, _ = seeded_subset_indices(targets, 1, 10, seed=0)
    assert na == 10
    assert (targets[a] == 1).all(), "must not leak other classes"
    assert not torch.equal(a.sort().values, b.sort().values), "subset must move with the seed"
    assert torch.equal(a, c), "same seed must give the same subset"


def test_seeded_subset_indices_caps_and_handles_none():
    from sweep_utils import seeded_subset_indices
    targets = torch.arange(30) % 3
    _, n = seeded_subset_indices(targets, 1, 10**6, seed=0)
    assert n == 10
    _, n = seeded_subset_indices(targets, 1, None, seed=0)
    assert n == 10
    _, n = seeded_subset_indices(targets, None, None, seed=0)
    assert n == 30, "label=None must keep every class"


def test_seeded_subset_indices_rejects_absent_label():
    from sweep_utils import seeded_subset_indices
    with pytest.raises(ValueError, match="no samples found"):
        seeded_subset_indices(torch.zeros(10, dtype=torch.long), 7, 3, 0)
