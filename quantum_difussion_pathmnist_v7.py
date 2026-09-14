import argparse
import os
import json
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
# `bottlenecks` lives next to this script; put that directory on sys.path so the
# import also works when this file is loaded by path (generate_samples.py, tests).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bottlenecks import ARMS, build_bottleneck, count_parameters
from sweep_utils import seeded_subset_indices
import pennylane as qml
from medmnist import PathMNIST
import torchvision.utils as vutils

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# === Cosine Beta Schedule ===
def cosine_beta_schedule(timesteps, s=0.008):
    steps = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
    alphas_cumprod = torch.cos(((steps / timesteps + s) / (1 + s)) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.999).float()


# === Timestep Embedding ===
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


# === Quantum Attention ===
class QuantumLayer(nn.Module):
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


# === UNet ===
class ImprovedUNet(nn.Module):
    """U-Net whose bottleneck gating module is chosen by `arm` (see bottlenecks.py).

    `use_quantum` is kept as a deprecated alias: True -> "quantum", False ->
    "plain". Note "plain" has no gating module at all and so is not a
    parameter-matched control for "quantum" -- use "se" for that (issue #3).
    """

    def __init__(self, channels=3, use_quantum=False, arm=None, n_hidden=16,
                 **quantum_kwargs):
        super().__init__()
        if arm is None:
            arm = "quantum" if use_quantum else "plain"
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
        self.arm = arm
        self.use_quantum = (arm == "quantum")
        self.enc1 = nn.Conv2d(channels, 32, 3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.res_block = ResBlock(128)
        self.time_embed_proj = nn.Linear(128, 128)
        # This script's QuantumLayer has a trainable input_proj, so the
        # down-projection is a Linear and all four arms apply.
        bottleneck = build_bottleneck(arm, QuantumLayer, channels=128,
                                     n_hidden=n_hidden, down="linear",
                                     **quantum_kwargs)
        if bottleneck is not None:
            self.q_attn = bottleneck
        self.dec1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.out = nn.Conv2d(64, channels, 3, padding=1)

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


# === Diffusion Process ===
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


# === Sample image ===
@torch.no_grad()
def sample(model, diffusion, steps=1000, n=1):
    """Return [n, 3, 28, 28] in [-1, 1]. `n` samples are drawn in one batch."""
    model.eval()
    x = torch.randn(n, 3, 28, 28, device=DEVICE)
    for t in reversed(range(steps)):
        t_tensor = torch.full((n,), t, device=DEVICE, dtype=torch.long)
        x = diffusion.p_sample(model, x, t_tensor)
    return x


# === Dataset ===
def load_pathmnist(label_condition=1, n_train=None, seed=0, batch_size=64):
    """DataLoader over one PathMNIST class, optionally a seeded subset of it.

    The label filter reads `dataset.labels` directly. It previously did
    `[i for i, (_, label) in enumerate(dataset) ...]`, which pulls every one of
    the ~90k training images through Resize + ToTensor + Normalize just to read
    its label. Measured on this dataset: 4.7 s versus 0.4 ms for an identical
    index (verified equal, not just equal in length).
    """
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])
    dataset = PathMNIST(split='train', download=True, transform=transform)

    labels = torch.as_tensor(dataset.labels).squeeze(-1)
    indices, n_used = seeded_subset_indices(labels, label_condition, n_train, seed)
    if label_condition is not None or n_train is not None:
        dataset = torch.utils.data.Subset(dataset, indices.tolist())

    loader = DataLoader(dataset, batch_size=min(batch_size, n_used), shuffle=True)
    return loader, n_used



# === Training ===
def train_pipeline(use_quantum=False, arm=None, save_dir="diffusion_pathmnist",
                   label_condition=1, n_train=None, seed=0, n_hidden=16,
                   batch_size=64, max_steps=None, epochs=30, sample_every=0,
                   device_name="default.qubit", diff_method="backprop"):
    """Train one arm on one PathMNIST class.

    Args:
        arm: one of bottlenecks.ARMS; overrides `use_quantum`.
        n_train: training images (issue #5's x-axis). None = the whole class.
        seed: seeds init, noise *and* the data subset.
        max_steps: budget in optimizer steps. Required for comparisons across
            n_train -- an epoch at N=10 is a single gradient step.
        sample_every: also dump samples every N steps (0 = only at the end).
            Sampling runs 1000 reverse steps, so the old every-epoch behaviour
            was a large part of the runtime.
    """
    if arm is None:
        arm = "quantum" if use_quantum else "plain"
    torch.manual_seed(seed)

    run_name = (f"path_{label_condition}_{arm}_n{n_train if n_train is not None else 'all'}"
                f"_s{seed}_t{max_steps if max_steps is not None else f'e{epochs}'}")
    save_dir = os.path.join(save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    loader, n_used = load_pathmnist(label_condition=label_condition, n_train=n_train,
                                   seed=seed, batch_size=batch_size)
    quantum_kwargs = ({"device_name": device_name, "diff_method": diff_method}
                      if arm == "quantum" else {})
    model = ImprovedUNet(channels=3, arm=arm, n_hidden=n_hidden,
                         **quantum_kwargs).to(DEVICE)
    bn_trainable, bn_total = model.bottleneck_parameter_counts()
    manifest = {
        "script": "pathmnist", "arm": arm, "label": label_condition,
        "n_train": n_used, "seed": seed, "batch_size": loader.batch_size,
        "n_hidden": n_hidden,
        "qdevice": device_name if arm == "quantum" else None,
        "diff_method": diff_method if arm == "quantum" else None,
        "model_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "bottleneck_trainable_params": bn_trainable,
        "bottleneck_total_params": bn_total,
    }
    with open(os.path.join(save_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[{run_name}] {json.dumps(manifest)}")
    # Built and loaded rather than deepcopy-ed: lightning.qubit holds a C++
    # StateVectorC128 that cannot be pickled.
    ema_model = ImprovedUNet(channels=3, arm=arm, n_hidden=n_hidden,
                             **quantum_kwargs).to(DEVICE)
    ema_model.load_state_dict(model.state_dict())
    diffusion = ImprovedGaussianDiffusion()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    steps_per_epoch = max(1, len(loader))
    total_steps = max_steps if max_steps is not None else epochs * steps_per_epoch
    manifest["steps_per_epoch"] = steps_per_epoch
    manifest["total_steps"] = total_steps
    with open(os.path.join(save_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

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
        # Checkpoint on the window mean, not on a single batch: best_model.pth is
        # what generate_samples.py loads, and a one-batch loss is far too noisy to
        # select with.
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

    # Save loss
    with open(os.path.join(save_dir, "loss.csv"), "w") as f:
        f.write("step,mean_loss\n")
        for step_i, l in loss_history:
            f.write(f"{step_i},{l}\n")

    plt.figure()
    plt.plot([st for st, _ in loss_history], [l for _, l in loss_history])
    plt.title(f"Training Loss ({run_name})")
    plt.xlabel("Step")
    plt.ylabel("Mean MSE Loss (window)")
    plt.grid(True)
    plt.savefig(f"{save_dir}/loss_curve.png")
    plt.close()


def dump_samples(ema_model, diffusion, save_dir, step, n=5):
    """Save the sampler's raw output (.pt) plus a contact sheet for eyeballing.

    The .pt is what evaluate.py reads. Do NOT compute metrics from the rendered
    figure -- make_grid's normalize/value_range and the rasterization both alter
    the pixels before measurement (issue #4).
    """
    batch = sample(ema_model, diffusion, n=n).cpu()
    torch.save(batch, f"{save_dir}/step{step:06d}_samples.pt")
    grid = vutils.make_grid(list(batch), nrow=n, normalize=True, value_range=(-1, 1))
    plt.figure(figsize=(10, 2))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/step{step:06d}_samples.png")
    plt.close()


def main():
    ap = argparse.ArgumentParser(
        description="Train the PathMNIST diffusion model, one arm of the #3 "
                    "ablation ladder at a time.")
    ap.add_argument("--arm", choices=ARMS, default="quantum",
                    help="'se' is the parameter-matched classical control; "
                         "'plain' has no gating module at all.")
    ap.add_argument("--label", type=int, default=1,
                    help="PathMNIST class to train on (None-like: use --all-labels)")
    ap.add_argument("--all-labels", action="store_true",
                    help="train on every class at once instead of one")
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="budget in optimizer steps; preferred over --epochs for "
                         "any comparison across --n-train")
    ap.add_argument("--epochs", type=int, default=30,
                    help="used only when --max-steps is omitted")
    ap.add_argument("--sample-every", type=int, default=0,
                    help="also dump samples every N steps (0 = only at the end)")
    ap.add_argument("--n-hidden", type=int, default=16)
    ap.add_argument("--qdevice", default="default.qubit",
                    help="lightning.qubit with --diff-method adjoint is ~3.5x faster")
    ap.add_argument("--diff-method", default="backprop",
                    choices=["backprop", "adjoint", "parameter-shift"])
    ap.add_argument("--save-dir", default=None, help="default: diffusion_pathmnist_<arm>")
    args = ap.parse_args()

    train_pipeline(arm=args.arm,
                   save_dir=args.save_dir or f"diffusion_pathmnist_{args.arm}",
                   label_condition=None if args.all_labels else args.label,
                   n_train=args.n_train, seed=args.seed, n_hidden=args.n_hidden,
                   batch_size=args.batch_size, max_steps=args.max_steps,
                   epochs=args.epochs, sample_every=args.sample_every,
                   device_name=args.qdevice, diff_method=args.diff_method)


if __name__ == '__main__':
    main()
