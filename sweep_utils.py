"""Shared pieces for the data-scaling sweep in issue #5.

The subsetting semantics live here rather than in each training script because
they are the part that has to be identical across scripts for a sweep to mean
anything: the subset must depend on the seed, so that the variance across seeds
includes *which images were drawn*. At N=10 that term dominates everything else,
so a sweep whose seeds only move the weight init would badly understate its own
error bars.
"""

import torch


def seeded_subset_indices(targets, label, n_train, seed):
    """Indices of `label` in `targets`, narrowed to a seeded subset of `n_train`.

    Args:
        targets: 1-D tensor (or anything tensor-convertible) of labels.
        label: the class to keep. None keeps every class.
        n_train: subset size. None, or >= the class size, keeps the whole class.
        seed: selects which subset is drawn.

    Returns:
        (indices LongTensor, n_used int)
    """
    targets = torch.as_tensor(targets)
    idx = torch.arange(len(targets)) if label is None else (targets == label).nonzero(as_tuple=True)[0]
    available = len(idx)
    if available == 0:
        raise ValueError(f"no samples found for label {label!r}")
    if n_train is None or n_train >= available:
        if n_train is not None and n_train > available:
            print(f"warning: asked for n_train={n_train} but label {label} has only "
                  f"{available} samples; using all of them")
        return idx, available
    g = torch.Generator().manual_seed(seed)
    pick = torch.randperm(available, generator=g)[:n_train]
    return idx[pick], n_train
