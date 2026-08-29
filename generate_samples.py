"""Generate samples from a trained checkpoint and save them as a raw tensor.

FID/KID need thousands of samples, and they need the model's actual output --
not an image recovered from a saved matplotlib figure.  This script produces a
single ``[N, C, 28, 28]`` float tensor in [-1, 1], which ``evaluate.py`` consumes.

Examples
--------
    python generate_samples.py --arch v7_mnist \\
        --checkpoint improved_diffusion_quantum/mnist_3/best_model.pth \\
        --n 10000 --out samples/quantum_mnist3.pt

    python generate_samples.py --arch v8_mnist --use-quantum \\
        --checkpoint diffusion_models_v8_quantum/mnist_3/best_model.pth \\
        --n 10000 --steps 1000 --batch-size 256 --out samples/v8_q_mnist3.pt
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent

ARCHS = {
    "v7_mnist": (REPO_ROOT / "quantum_difussion_mnist_v7.py", 1),
    "v7_pathmnist": (REPO_ROOT / "quantum_difussion_pathmnist_v7.py", 3),
    "v8_mnist": (REPO_ROOT / "full_unet" / "quantum_diffusion_mnist_v8.py", 1),
}


def load_arch(name):
    """Import one of the training scripts by path (they are not a package)."""
    path, channels = ARCHS[name]
    spec = importlib.util.spec_from_file_location(f"_arch_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, channels


def build_model(module, arch, use_quantum, device):
    if arch == "v7_pathmnist":
        model = module.ImprovedUNet(channels=3, use_quantum=use_quantum)
    else:
        model = module.ImprovedUNet(use_quantum=use_quantum)
    return model.to(device).eval()


@torch.no_grad()
def generate(model, diffusion, n, channels, steps, batch_size, device):
    """Run the reverse process in batches and return an [n, C, 28, 28] tensor."""
    out = []
    remaining = n
    with tqdm(total=n, desc="sampling") as bar:
        while remaining > 0:
            b = min(batch_size, remaining)
            x = torch.randn(b, channels, 28, 28, device=device)
            for t in reversed(range(steps)):
                t_tensor = torch.full((b,), t, device=device, dtype=torch.long)
                x = diffusion.p_sample(model, x, t_tensor)
            out.append(x.cpu())
            remaining -= b
            bar.update(b)
    return torch.cat(out)[:n]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", required=True, choices=sorted(ARCHS))
    p.add_argument("--checkpoint", required=True, type=Path,
                   help="state_dict saved by train_pipeline (e.g. best_model.pth)")
    p.add_argument("--out", required=True, type=Path, help="destination .pt file")
    p.add_argument("--n", type=int, default=10000,
                   help="number of samples (default: 10000; FID is badly biased below ~10k)")
    p.add_argument("--steps", type=int, default=1000, help="reverse diffusion steps")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--use-quantum", action="store_true",
                   help="must match the checkpoint's architecture")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="default: cuda if available")
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)

    module, channels = load_arch(args.arch)
    model = build_model(module, args.arch, args.use_quantum, device)
    state = torch.load(args.checkpoint, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        # Loudly, rather than silently evaluating a partly random model.
        raise SystemExit(
            f"checkpoint does not match the model.\n"
            f"  missing keys:    {sorted(missing)}\n"
            f"  unexpected keys: {sorted(unexpected)}\n"
            f"Check --arch and --use-quantum."
        )

    diffusion = module.ImprovedGaussianDiffusion(timesteps=args.steps)
    samples = generate(model, diffusion, args.n, channels, args.steps,
                       args.batch_size, device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(samples, args.out)
    print(f"saved {tuple(samples.shape)} to {args.out} "
          f"(range [{samples.min():.3f}, {samples.max():.3f}])")


if __name__ == "__main__":
    main()
