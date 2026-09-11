# === Improved FlexibleDiffusionUNet (added clip-based sampling + cosine scheduling + more accurate EMA + optional quantum attention) ===

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import copy
import json

# `bottlenecks` lives next to this script. Put that directory on sys.path so the
# import also works when this file is loaded by path from somewhere else, which
# generate_samples.py and the test suite both do.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bottlenecks import ARMS, build_bottleneck, count_parameters
import pennylane as qml

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Cosine Beta Schedule ===
def cosine_beta_schedule(timesteps, s=0.008):
    steps = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
    alphas_cumprod = torch.cos(((steps / timesteps + s) / (1 + s)) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.999).float()

# === Sinusoidal Timestep Embedding ===
def get_timestep_embedding(timesteps, embedding_dim=128):
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    return torch.cat([emb.sin(), emb.cos()], dim=1)

# === ResBlock ===
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        return F.relu(x + self.block(x))

# === Quantum Attention Layer ===
class QuantumLayer(nn.Module):
    """Variational circuit as a channel-gating bottleneck.

    `device_name`/`diff_method` are exposed because the default
    (`default.qubit` + `backprop`) is the slow combination: it backpropagates
    through the whole 2^n_qubits statevector. Measured at n_qubits=16,
    batch 64, one forward+backward on one CPU core of this machine:

        default.qubit   + backprop          21.97 s
        lightning.qubit + adjoint            6.22 s   (3.5x faster)

    The two agree to ~5e-07 on gradients of order 1 (checked against each other
    on an 8-qubit copy of this circuit), so `lightning.qubit` + `adjoint` is the
    better choice for any sweep -- see issue #5, where the grid is dominated by
    this cost. The default is left unchanged so existing runs stay bit-comparable.
    """

    def __init__(self, n_qubits=16, n_layers=3,
                 device_name="default.qubit", diff_method="backprop"):
        super().__init__()
        dev = qml.device(device_name, wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(inputs, weights):
            # `inputs[..., i]` (rather than `inputs[i]`) so the same circuit works for
            # a single sample and for a batched [B, n_qubits] input.
            for i in range(n_qubits):
                qml.RY(inputs[..., i], wires=i)
            for l in range(n_layers):
                for i in range(n_qubits):
                    # A circuit built only from computational-basis-diagonal gates (RZ)
                    # and computational-basis permutations (CNOT) cannot change
                    # |<b|psi>|^2, so every <PauliZ> would be independent of `weights`
                    # and d<Z>/dweights would be identically zero. The RY makes the
                    # RZ phases observable and restores trainability.
                    qml.RZ(weights[l, i, 0], wires=i)
                    qml.RY(weights[l, i, 1], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits, 2)}
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        self.input_proj = nn.Linear(128, n_qubits)
        self.output_proj = nn.Linear(n_qubits, 128)

    def forward(self, x):
        x = self.input_proj(x)
        # `.to('cpu')` rather than `.detach().cpu()`: detaching cut the quantum branch
        # out of the autograd graph on the input side, so `input_proj` (and the encoder
        # upstream of it) never received any gradient. The QNode still executes on CPU.
        out = self.qlayer(x.to('cpu')).to(x.device)
        return self.output_proj(out)

# === UNet with a selectable bottleneck arm (see bottlenecks.py / issue #3) ===
class ImprovedUNet(nn.Module):
    """U-Net whose bottleneck gating module is chosen by `arm`.

    `arm` is one of bottlenecks.ARMS. `use_quantum` is kept as a deprecated
    alias so existing callers and checkpoints keep working: True -> "quantum",
    False -> "plain". Note that "plain" has no gating module at all, which is
    why it is not a parameter-matched control for "quantum" -- use "se" for
    that.
    """

    def __init__(self, use_quantum=False, arm=None, n_hidden=16, **quantum_kwargs):
        super().__init__()
        if arm is None:
            arm = "quantum" if use_quantum else "plain"
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
        self.arm = arm
        # Kept so `model.use_quantum` still answers the old question.
        self.use_quantum = (arm == "quantum")
        self.enc1 = nn.Conv2d(1, 32, 3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.res_block = ResBlock(128)
        self.time_embed_proj = nn.Linear(128, 128)
        # `q_attn` keeps its name across all arms so checkpoints and the
        # existing tests stay addressable; for "plain" it is simply absent.
        bottleneck = build_bottleneck(arm, QuantumLayer, channels=128,
                                     n_hidden=n_hidden, **quantum_kwargs)
        if bottleneck is not None:
            self.q_attn = bottleneck
        self.dec1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.out = nn.Conv2d(64, 1, 3, padding=1)

    def bottleneck_parameter_counts(self):
        """(trainable, total) params in the gating module -- for the #3 table."""
        return count_parameters(getattr(self, "q_attn", None))

    def forward(self, x, t):
        t_embed = get_timestep_embedding(t, 128).to(x.device)
        t_proj = self.time_embed_proj(t_embed).unsqueeze(-1).unsqueeze(-1)
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(e1))
        e3 = F.relu(self.enc3(e2)) + t_proj
        x = self.res_block(e3)
        pooled = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
        if self.arm != "plain":
            q_weight = self.q_attn(pooled).unsqueeze(-1).unsqueeze(-1)
            x = x * q_weight
        d1 = F.relu(self.dec1(x))
        d2 = F.relu(self.dec2(d1))
        d2 = torch.cat([d2, e1], dim=1)
        return self.out(d2)

# === improved Gaussian Diffusion with clip ===
class ImprovedGaussianDiffusion:
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps
        self.beta = cosine_beta_schedule(timesteps).to(DEVICE)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus = torch.sqrt(1 - self.alpha_bar)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return self.sqrt_alpha_bar[t].view(-1, 1, 1, 1) * x_start + \
               self.sqrt_one_minus[t].view(-1, 1, 1, 1) * noise

    @torch.no_grad()
    def p_sample(self, model, x, t):
        noise_pred = model(x, t)
        beta_t = self.beta[t].view(-1, 1, 1, 1)
        alpha_t = self.alpha[t].view(-1, 1, 1, 1)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)

        x0_pred = (x - (1 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt()
        x0_pred = x0_pred.clamp(-1, 1)

        if t[0] > 0:
            alpha_bar_prev = self.alpha_bar[t - 1].view(-1, 1, 1, 1)
            mean = beta_t * alpha_bar_prev.sqrt() / (1 - alpha_bar_t) * x0_pred + \
                   (1 - alpha_bar_prev) * alpha_t.sqrt() / (1 - alpha_bar_t) * x
            std = (beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)).sqrt()
            noise = torch.randn_like(x)
            return mean + std * noise
        else:
            return x0_pred

# === Sampling ===
@torch.no_grad()
def sample(model, diffusion, steps=1000, n=1):
    """Return [n, 1, 28, 28] in [-1, 1]. `n` samples are drawn in one batch."""
    model.eval()
    x = torch.randn(n, 1, 28, 28, device=DEVICE)
    for t in reversed(range(steps)):
        t_tensor = torch.full((n,), t, device=DEVICE, dtype=torch.long)
        x = diffusion.p_sample(model, x, t_tensor)
    return x

# === Training ===
def subsample(dataset, digit_label, n_train, seed):
    """Keep only `digit_label`, then a seeded random subset of `n_train` images.

    The subset is resampled per seed on purpose (issue #5): at N=10 *which* ten
    images you draw dominates every other source of variance, so the seed has to
    move the data, not just the init.
    """
    mask = dataset.targets == digit_label
    dataset.data = dataset.data[mask]
    dataset.targets = dataset.targets[mask]
    available = len(dataset.data)
    if n_train is None or n_train >= available:
        if n_train is not None and n_train > available:
            print(f"warning: asked for n_train={n_train} but digit {digit_label} "
                  f"only has {available} images; using all of them")
        return dataset, available
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(available, generator=g)[:n_train]
    dataset.data = dataset.data[idx]
    dataset.targets = dataset.targets[idx]
    return dataset, n_train


def train_pipeline(digit_label=1, use_quantum=False, arm=None,
                   save_dir="improved_diffusion_classical", n_train=None, seed=0,
                   max_steps=None, epochs=30, batch_size=64, sample_every=0,
                   n_hidden=16, lr=3e-4, device_name="default.qubit",
                   diff_method="backprop"):
    """Train one arm on one digit.

    Args:
        arm: one of bottlenecks.ARMS. Overrides `use_quantum` when given.
        n_train: images to train on (issue #5's x-axis). None = the whole class.
        seed: seeds init, noise, *and* the data subset.
        max_steps: budget in optimizer steps. The budget has to be in steps
            rather than epochs: at N=10 an "epoch" is one gradient step, so an
            epoch-based budget leaves the low-N arms undertrained rather than
            data-limited, which would confound the whole scaling study.
        sample_every: also dump samples every this many steps (0 = only at end).
            Sampling runs 1000 reverse steps, so this is expensive.
    """
    if arm is None:
        arm = "quantum" if use_quantum else "plain"
    torch.manual_seed(seed)

    run_name = f"mnist_{digit_label}_{arm}_n{n_train if n_train is not None else 'all'}_s{seed}"
    save_dir = os.path.join(save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataset, n_used = subsample(dataset, digit_label, n_train, seed)
    loader = DataLoader(dataset, batch_size=min(batch_size, n_used), shuffle=True, drop_last=False)

    quantum_kwargs = ({"device_name": device_name, "diff_method": diff_method}
                      if arm == "quantum" else {})
    model = ImprovedUNet(arm=arm, n_hidden=n_hidden, **quantum_kwargs).to(DEVICE)
    # Build the EMA copy rather than deepcopy-ing it: lightning.qubit holds a C++
    # StateVectorC128 that cannot be pickled, so copy.deepcopy(model) raises for
    # any non-default simulator. Constructing and loading the state dict is
    # equivalent and backend-agnostic.
    ema_model = ImprovedUNet(arm=arm, n_hidden=n_hidden, **quantum_kwargs).to(DEVICE)
    ema_model.load_state_dict(model.state_dict())
    diffusion = ImprovedGaussianDiffusion()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    steps_per_epoch = max(1, len(loader))
    total_steps = max_steps if max_steps is not None else epochs * steps_per_epoch
    bn_trainable, bn_total = model.bottleneck_parameter_counts()
    manifest = {
        "arm": arm, "digit": digit_label, "n_train": n_used, "seed": seed,
        "total_steps": total_steps, "batch_size": loader.batch_size,
        "steps_per_epoch": steps_per_epoch, "lr": lr, "n_hidden": n_hidden,
        "model_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "qdevice": device_name if arm == "quantum" else None,
        "diff_method": diff_method if arm == "quantum" else None,
        "bottleneck_trainable_params": bn_trainable,
        "bottleneck_total_params": bn_total,
    }
    with open(os.path.join(save_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[{run_name}] {json.dumps(manifest)}")

    loss_history = []   # (step, mean loss over the window)
    window = []
    best_loss = float('inf')

    def infinite(loader):
        while True:
            for batch in loader:
                yield batch

    stream = infinite(loader)
    bar = tqdm(range(1, total_steps + 1), desc=run_name)
    for step in bar:
        x, _ = next(stream)
        x = x.to(DEVICE)
        t = torch.randint(0, diffusion.timesteps, (x.size(0),), device=DEVICE).long()
        noise = torch.randn_like(x)
        x_noisy = diffusion.q_sample(x, t, noise)
        noise_pred = model(x_noisy, t)
        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                p_ema.data.mul_(0.999).add_(p.data, alpha=1 - 0.999)

        window.append(loss.item())
        # Checkpoint on the mean over the last epoch-equivalent, not on a single
        # batch: a one-batch loss is far too noisy to select a model with, and
        # `best_model.pth` is what generate_samples.py loads.
        if step % steps_per_epoch == 0 or step == total_steps:
            mean_loss = sum(window) / len(window)
            loss_history.append((step, mean_loss))
            bar.set_postfix(mean_loss=f"{mean_loss:.4f}")
            window = []
            if mean_loss < best_loss:
                best_loss = mean_loss
                torch.save(ema_model.state_dict(), os.path.join(save_dir, "best_model.pth"))

        if sample_every and step % sample_every == 0:
            dump_samples(ema_model, diffusion, save_dir, step)

    torch.save(ema_model.state_dict(), os.path.join(save_dir, "final_model.pth"))
    dump_samples(ema_model, diffusion, save_dir, total_steps)

    with open(os.path.join(save_dir, "loss.csv"), "w") as f:
        f.write("step,mean_loss\n")
        for step, l in loss_history:
            f.write(f"{step},{l}\n")

    plt.plot([s for s, _ in loss_history], [l for _, l in loss_history])
    plt.title(f"Training Loss ({run_name})")
    plt.xlabel("Step")
    plt.ylabel("MSE Loss (window mean)")
    plt.savefig(f"{save_dir}/loss_curve.png")
    plt.close()

    manifest["best_mean_loss"] = best_loss
    with open(os.path.join(save_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def dump_samples(ema_model, diffusion, save_dir, step, n=5):
    """Save the sampler's raw output (.pt) plus a contact sheet for eyeballing.

    The .pt file is what evaluate.py reads. Do NOT compute metrics from the .png:
    imshow rescales each subplot to its own min/max and the figure is rasterized
    at the figure DPI, so intensity and geometry are both altered (issue #4).
    """
    imgs = sample(ema_model, diffusion, steps=1000, n=n)
    torch.save(imgs.cpu(), f"{save_dir}/step{step:06d}_samples.pt")
    plt.figure(figsize=(10, 2))
    for i in range(imgs.size(0)):
        plt.subplot(1, n, i + 1)
        plt.imshow(imgs[i].squeeze().cpu().numpy(), cmap='gray')
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/step{step:06d}_samples.png")
    plt.close()


def main():
    ap = argparse.ArgumentParser(
        description="Train the v7 MNIST diffusion model, one arm of the #3 "
                    "ablation ladder at a time.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  # the honest quantum-vs-classical pair at N=100
  python quantum_difussion_mnist_v7.py --arm quantum --digits 3 --n-train 100 --max-steps 2000
  python quantum_difussion_mnist_v7.py --arm se      --digits 3 --n-train 100 --max-steps 2000

  # all four arms, one digit, 5 seeds (issue #5's grid, one N)
  for a in plain se se_frozen quantum; do for s in 0 1 2 3 4; do
    python quantum_difussion_mnist_v7.py --arm $a --digits 3 --n-train 100 \
        --max-steps 2000 --seed $s --save-dir runs
  done; done
""")
    ap.add_argument("--arm", choices=ARMS, default="quantum",
                    help="bottleneck arm (default: quantum). 'se' is the "
                         "parameter-matched classical control; 'plain' has no "
                         "gating module at all and is NOT a matched control.")
    ap.add_argument("--digits", type=int, nargs="+", default=list(range(10)),
                    help="MNIST classes to train, one run each (default: 0-9). "
                         "Note the previous hard-coded loop was range(1, 10), so "
                         "digit 0 was never trained by this script.")
    ap.add_argument("--n-train", type=int, default=None,
                    help="training images per class; omit for the whole class "
                         "(~5400-6700). This is issue #5's x-axis.")
    ap.add_argument("--seed", type=int, default=0,
                    help="seeds init, noise and the data subset")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="budget in optimizer steps. Strongly preferred over "
                         "--epochs for any comparison across --n-train, since an "
                         "epoch at N=10 is a single gradient step.")
    ap.add_argument("--epochs", type=int, default=30,
                    help="used only when --max-steps is omitted")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-hidden", type=int, default=16,
                    help="bottleneck width; the quantum arm's qubit count")
    ap.add_argument("--sample-every", type=int, default=0,
                    help="also dump samples every N steps (0 = only at the end). "
                         "Each dump runs 1000 reverse steps.")
    ap.add_argument("--qdevice", default="default.qubit",
                    help="PennyLane device for the quantum arm. lightning.qubit "
                         "with --diff-method adjoint is ~3.5x faster at n_qubits=16 "
                         "and agrees with the default to ~5e-07 on gradients.")
    ap.add_argument("--diff-method", default="backprop",
                    choices=["backprop", "adjoint", "parameter-shift"],
                    help="differentiation method for the quantum arm "
                         "(adjoint requires lightning.qubit)")
    ap.add_argument("--save-dir", default=None,
                    help="default: improved_diffusion_<arm>")
    args = ap.parse_args()

    save_dir = args.save_dir or f"improved_diffusion_{args.arm}"
    for digit in args.digits:
        train_pipeline(digit_label=digit, arm=args.arm, save_dir=save_dir,
                       n_train=args.n_train, seed=args.seed,
                       max_steps=args.max_steps, epochs=args.epochs,
                       batch_size=args.batch_size, sample_every=args.sample_every,
                       n_hidden=args.n_hidden, lr=args.lr,
                       device_name=args.qdevice, diff_method=args.diff_method)


if __name__ == '__main__':
    main()
