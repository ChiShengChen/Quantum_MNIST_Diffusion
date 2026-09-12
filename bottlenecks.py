"""The four arms of the ablation ladder in issue #3.

`use_quantum=True` used to add a module with no counterpart at all in the
`use_quantum=False` baseline, so the comparison measured "quantum circuit" and
"channel-wise gating exists at all" together. The quantum arm is structurally a
Squeeze-and-Excitation block (Hu et al., CVPR 2018) with reduction ratio 8 and a
QNode where SE has its nonlinearity, so the classical control has to be that
same block.

    arm           bottleneck                                  isolates
    ----------------------------------------------------------------------------
    plain         none                                        the old baseline
    se            Linear(C,h) -> ReLU -> Linear(h,C)           "does gating help?"
    se_frozen     same, down-projection frozen at init         "is it just a random
                                                                projection?"
    quantum       Linear(C,h) -> QNode -> Linear(h,C)          the circuit itself

Which arms apply depends on the script: v7 and the pathmnist model have a
trainable `input_proj` inside `QuantumLayer`, so all four arms are available. v8's
circuit pools its own input instead, so there is no down-projection to freeze and
`se_frozen` is rejected there rather than silently duplicating `se`.

`se` vs `quantum` is the honest test of the circuit: the two differ only by the
QNode in place of the ReLU, plus the circuit's own parameters.

`se_frozen` exists because of the `.detach()` bug (#2): it reproduces the
"down-projection is a fixed random projection" condition that the buggy quantum
arm was actually training under, so `quantum ~= se_frozen` would mean the
measured gain came from the random projection, not the circuit.

The gate is applied as a plain multiplication, matching what the quantum arm
already did (`x = x * q_weight`) -- deliberately *not* a sigmoid, so the arms
differ only in the bottleneck and not in how the gate is bounded.
"""

import torch.nn as nn
import torch.nn.functional as F

ARMS = ("plain", "se", "se_frozen", "quantum")


class SEBottleneck(nn.Module):
    """Classical counterpart of ``QuantumLayer``.

    Mirrors the quantum arm's shape exactly: reduce ``channels -> n_hidden``,
    apply a nonlinearity, project back. ``n_hidden`` defaults to the quantum
    arm's qubit count so the two are parameter-matched up to the circuit's own
    weights.

    Args:
        channels: width of the feature vector being gated.
        n_hidden: bottleneck width (the quantum arm's ``n_qubits``).
        freeze_down: if True, the down-projection is frozen at initialization
            (the ``se_frozen`` arm). Only meaningful when ``down="linear"``.
        down: how the reduction to ``n_hidden`` is done, matching whichever
            ``QuantumLayer`` this arm is standing in for:

            * ``"linear"`` -- ``nn.Linear(channels, n_hidden)``, as in v7 and
              the pathmnist script, whose QuantumLayer has an ``input_proj``.
            * ``"pool"`` -- ``adaptive_avg_pool1d``, as in v8, whose circuit
              reduces its own input and therefore has **no** down-projection
              parameters. Using ``"linear"`` against v8 would hand the classical
              arm ``channels*n_hidden + n_hidden`` parameters the quantum arm
              does not have, which is the confound this ladder exists to remove.
    """

    def __init__(self, channels=128, n_hidden=16, freeze_down=False, down="linear"):
        super().__init__()
        if down not in ("linear", "pool"):
            raise ValueError(f"unknown down={down!r}; expected 'linear' or 'pool'")
        if down == "pool" and freeze_down:
            raise ValueError(
                "freeze_down is meaningless with down='pool': adaptive_avg_pool1d "
                "has no parameters to freeze, so the arm would be identical to "
                "'se'. The se_frozen arm only applies where QuantumLayer has a "
                "trainable input_proj (v7 / pathmnist), not to v8."
            )
        self.down_mode = down
        self.n_hidden = n_hidden
        self.down = nn.Linear(channels, n_hidden) if down == "linear" else None
        self.act = nn.ReLU()
        self.up = nn.Linear(n_hidden, channels)
        self.freeze_down = freeze_down
        if freeze_down:
            for p in self.down.parameters():
                p.requires_grad_(False)

    def reduce(self, x):
        if self.down_mode == "linear":
            return self.down(x)
        # Matches v8's circuit, which pools its own input down to n_qubits.
        return F.adaptive_avg_pool1d(x.unsqueeze(1), self.n_hidden).squeeze(1)

    def forward(self, x):
        return self.up(self.act(self.reduce(x)))


def build_bottleneck(arm, quantum_layer_cls, channels=128, n_hidden=16,
                     down="linear", **quantum_kwargs):
    """Return the bottleneck module for `arm`, or None for the plain arm.

    `quantum_layer_cls` is passed in rather than imported so this module stays
    usable from v7, v8 and the pathmnist script, each of which defines its own
    ``QuantumLayer``.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    if arm == "plain":
        return None
    if arm == "quantum":
        # quantum_kwargs carries simulator choices (device_name, diff_method);
        # the classical arms have no use for them.
        return quantum_layer_cls(n_qubits=n_hidden, **quantum_kwargs)
    return SEBottleneck(channels, n_hidden, freeze_down=(arm == "se_frozen"), down=down)


def count_parameters(module):
    """(trainable, total) parameter counts; None-safe for the plain arm."""
    if module is None:
        return 0, 0
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return trainable, total
