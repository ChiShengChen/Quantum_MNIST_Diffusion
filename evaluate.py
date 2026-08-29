"""Compute FID and KID for generated samples against a real dataset.

This replaces the ``cal_fid_ssim*.py`` route, which recovered "generated images"
by cropping them back out of a saved matplotlib figure.  That round trip
(per-axes min/max rescaling in ``imshow`` -> DPI rasterization -> PNG -> global
threshold -> column-gap segmentation -> bicubic resize) is lossy and applies to
the generated images only, so the real and generated distributions differ by an
entire image-processing pipeline before Inception ever sees them.  It also
over-segmented: the committed CSVs report 13 "digits" recovered from figures
containing 5 samples, and a 2048x2048 covariance estimated from 13 vectors has
rank <= 12.

Here both sides are uint8 in [0, 255] and reach the feature extractor by the
same path.

Notes on the metrics
--------------------
* **FID** is biased at small N; the bias is severe in the low-data regime this
  project cares about.  Use >= 10k samples, or read KID instead.
* **KID** is an unbiased estimator and is far more stable at moderate N, so it
  is reported alongside FID and is the one to trust for small sample counts.
* **SSIM is deliberately not computed.**  The previous scripts paired generated
  sample *i* with real image *i* in dataset order; SSIM between two unrelated
  samples has no relationship to generative quality, and averaging it does not
  make it one.  Use precision/recall or density/coverage if you need a
  fidelity-vs-diversity split.

Examples
--------
    python evaluate.py --generated samples/quantum_mnist3.pt --dataset mnist --label 3
    python evaluate.py --generated samples/q.pt --dataset pathmnist --label 1 --n-real 10000
"""

import argparse
import json
from pathlib import Path

import torch


def to_uint8(x):
    """[-1, 1] float tensor -> uint8 [0, 255], replicated to 3 channels."""
    if x.dim() != 4:
        raise ValueError(f"expected [N, C, H, W], got {tuple(x.shape)}")
    x = ((x.clamp(-1.0, 1.0) + 1.0) / 2.0 * 255.0).round().to(torch.uint8)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    if x.shape[1] != 3:
        raise ValueError(f"expected 1 or 3 channels, got {x.shape[1]}")
    return x


def load_real_mnist(label, n, root="./data"):
    """Real MNIST as uint8, straight from the dataset (no float round trip)."""
    from torchvision.datasets import MNIST

    ds = MNIST(root=root, train=True, download=True)
    data = ds.data if label is None else ds.data[ds.targets == label]
    if len(data) < n:
        print(f"warning: only {len(data)} real images available for label {label}")
    return data[:n].unsqueeze(1).repeat(1, 3, 1, 1).contiguous()


def load_real_pathmnist(label, n):
    from medmnist import PathMNIST

    ds = PathMNIST(split="train", download=True)
    imgs = torch.from_numpy(ds.imgs)  # [N, 28, 28, 3] uint8
    if label is not None:
        labels = torch.from_numpy(ds.labels).squeeze(-1)
        imgs = imgs[labels == label]
    if len(imgs) < n:
        print(f"warning: only {len(imgs)} real images available for label {label}")
    return imgs[:n].permute(0, 3, 1, 2).contiguous()


def compute_metrics(generated, real, device, kid_subset_size=None, feature=2048):
    """FID and KID between two uint8 image tensors."""
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance

    if kid_subset_size is None:
        # KID's default subset_size of 1000 raises if either set is smaller.
        kid_subset_size = max(2, min(1000, len(generated), len(real)))

    fid = FrechetInceptionDistance(feature=feature, normalize=False).to(device)
    kid = KernelInceptionDistance(feature=feature, subset_size=kid_subset_size,
                                  normalize=False).to(device)

    for metric in (fid, kid):
        _feed(metric, real, device, real=True)
        _feed(metric, generated, device, real=False)

    kid_mean, kid_std = kid.compute()
    return {
        "fid": float(fid.compute()),
        "kid_mean": float(kid_mean),
        "kid_std": float(kid_std),
        "n_generated": int(len(generated)),
        "n_real": int(len(real)),
        "kid_subset_size": int(kid_subset_size),
    }


def _feed(metric, images, device, real, batch_size=64):
    for i in range(0, len(images), batch_size):
        metric.update(images[i : i + batch_size].to(device), real=real)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generated", required=True, type=Path,
                   help=".pt file written by generate_samples.py (float, [-1, 1])")
    p.add_argument("--dataset", required=True, choices=["mnist", "pathmnist"])
    p.add_argument("--label", type=int, default=None,
                   help="class to compare against; omit to use the whole training split")
    p.add_argument("--n-real", type=int, default=10000)
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    generated = to_uint8(torch.load(args.generated, map_location="cpu"))
    if args.dataset == "mnist":
        real = load_real_mnist(args.label, args.n_real)
    else:
        real = load_real_pathmnist(args.label, args.n_real)

    if len(generated) < 10000:
        print(f"warning: {len(generated)} generated samples. FID is heavily biased "
              f"below ~10k; prefer KID at this sample count.")

    results = compute_metrics(generated, real, device)
    results.update(generated=str(args.generated), dataset=args.dataset, label=args.label)

    print(json.dumps(results, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
