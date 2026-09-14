# === Quantum Diffusion MNIST v8 (Improved UNet, ResBlocks, Time Embedding) ===

import argparse
import os
import json
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
# `bottlenecks` lives in the repo root, one level up from full_unet/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bottlenecks import ARMS, build_bottleneck, count_parameters
from sweep_utils import seeded_subset_indices
import pennylane as qml

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TIME_EMBED_DIM = 128 # Define time embedding dimension globally

# === Cosine Beta Schedule ===
def cosine_beta_schedule(timesteps, s=0.008):
    steps = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
    alphas_cumprod = torch.cos(((steps / timesteps + s) / (1 + s)) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.999).float()

# === Sinusoidal Timestep Embedding ===
def get_timestep_embedding(timesteps, embedding_dim=TIME_EMBED_DIM):
    assert embedding_dim % 2 == 0
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    return torch.cat([emb.sin(), emb.cos()], dim=1)

# === ResBlock with Time Embedding ===
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=TIME_EMBED_DIM):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU()
        # Ensure dimensions match for residual connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = self.relu(self.conv1(x))
        time_out = self.relu(self.time_mlp(t_emb))
        # Add time embedding (broadcast): [B, C] -> [B, C, 1, 1]
        h = h + time_out.unsqueeze(-1).unsqueeze(-1)
        h = self.relu(self.conv2(h))
        return h + self.shortcut(x) # Residual connection

# === Quantum Attention Layer ===
# (Unchanged from v7, but ensures input/output matches TIME_EMBED_DIM if used)
class QuantumLayer(nn.Module):
    def __init__(self, n_qubits=16, n_layers=3, embed_dim=TIME_EMBED_DIM,
                 device_name="default.qubit", diff_method="backprop"):
        super().__init__()
        self.n_qubits = n_qubits
        dev = qml.device(device_name, wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(inputs, weights):
            # Ensure input features match n_qubits
            inputs_resized = F.adaptive_avg_pool1d(inputs.unsqueeze(1), n_qubits).squeeze(1)
            for i in range(n_qubits):
                # Scale inputs to typical angle range like [0, pi]
                qml.RY(torch.pi * torch.sigmoid(inputs_resized[:, i]), wires=i)
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
                # Optional: Entangling layer like CNOT ladder or ring
                # qml.CNOT(wires=[n_qubits - 1, 0]) # Example: cycle entanglement
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits, 2)}
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        # Projects final qubit measurements back to the embedding dimension
        self.output_proj = nn.Linear(n_qubits, embed_dim)

    def forward(self, x):
        # x shape: [Batch, embed_dim]
        # No input projection needed if embed_dim is handled correctly upstream
        # `.to('cpu')` rather than `.detach().cpu()`: detaching cut the quantum branch
        # out of the autograd graph on the input side, so the encoder never received
        # any gradient through it. The QNode still executes on CPU.
        x_cpu = x.to('cpu') # QNode execution on CPU
        quantum_output = self.qlayer(x_cpu) # Shape: [Batch, n_qubits]
        quantum_output = quantum_output.to(x.device) # Move back to original device
        return self.output_proj(quantum_output) # Project back to embed_dim


# === Improved UNet with ResBlocks, Time Embedding, and Skip Connections ===
class ImprovedUNet(nn.Module):
    def __init__(self, use_quantum=False, time_embed_dim=TIME_EMBED_DIM,
                 arm=None, n_hidden=16, **quantum_kwargs):
        """`arm` selects the bottleneck module; see bottlenecks.py and issue #3.

        Unlike v7, this model's QuantumLayer reduces its own input with
        `adaptive_avg_pool1d`, so the quantum arm has **no** down-projection
        parameters. The classical control therefore has to pool too
        (`down="pool"`), and `se_frozen` does not apply here -- there is nothing
        to freeze. Passing it raises rather than silently duplicating `se`.
        """
        super().__init__()
        if arm is None:
            arm = "quantum" if use_quantum else "plain"
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
        self.arm = arm
        self.use_quantum = (arm == "quantum")
        self.time_embed_dim = time_embed_dim

        # Initial projection
        self.init_conv = nn.Conv2d(1, 32, 3, padding=1)

        # Encoder
        self.enc_res1 = ResBlock(32, 32, time_embed_dim)
        self.down1 = nn.Conv2d(32, 64, 3, stride=2, padding=1) # Downsample
        self.enc_res2 = ResBlock(64, 64, time_embed_dim)
        self.down2 = nn.Conv2d(64, 128, 3, stride=2, padding=1) # Downsample
        self.enc_res3 = ResBlock(128, 128, time_embed_dim)

        # Bottleneck (Optional Quantum Attention)
        self.mid_res = ResBlock(128, 128, time_embed_dim)
        # The bottleneck feature dim is 128. down="pool" mirrors this script's
        # QuantumLayer, which pools 128 -> n_qubits inside the circuit rather
        # than through a trainable Linear; a Linear here would hand the classical
        # arm 2064 parameters the quantum arm does not have.
        bottleneck = build_bottleneck(arm, QuantumLayer, channels=128,
                                     n_hidden=n_hidden, down="pool",
                                     **quantum_kwargs)
        if bottleneck is not None:
            self.q_attn = bottleneck

        # Decoder
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2) # Upsample
        # Decoder ResBlock takes concatenated input (skip + upsampled)
        self.dec_res1 = ResBlock(64 + 64, 64, time_embed_dim) # Corrected input channels
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2) # Upsample
        self.dec_res2 = ResBlock(32 + 32, 32, time_embed_dim) # Corrected input channels

        # Final Layer
        self.out = nn.Conv2d(32, 1, 1) # Use 1x1 conv for final projection

    def bottleneck_parameter_counts(self):
        """(trainable, total) params in the gating module -- for the #3 table."""
        return count_parameters(getattr(self, "q_attn", None))

    def forward(self, x, t):
        # Time embedding
        t_emb = get_timestep_embedding(t, self.time_embed_dim).to(x.device)

        # Encoder
        h1 = self.init_conv(x)
        h1 = self.enc_res1(h1, t_emb) # [B, 32, 28, 28]

        h2 = self.down1(h1)
        h2 = self.enc_res2(h2, t_emb) # [B, 64, 14, 14]

        h3 = self.down2(h2)
        h3 = self.enc_res3(h3, t_emb) # [B, 128, 7, 7]

        # Bottleneck
        mid = self.mid_res(h3, t_emb)

        if self.arm != "plain":
             # Apply the bottleneck gate on pooled features
            pooled = F.adaptive_avg_pool2d(mid, (1, 1)).squeeze(-1).squeeze(-1) # [B, 128]
            q_weight = self.q_attn(pooled) # Shape: [B, 128]
            # Apply as channel-wise scaling (needs unsqueezing)
            mid = mid * q_weight.unsqueeze(-1).unsqueeze(-1) # [B, 128, 7, 7]

        # Decoder
        d1 = self.up1(mid) # [B, 64, 14, 14]
        # Skip connection from h2
        d1 = torch.cat([d1, h2], dim=1) # [B, 64+64, 14, 14]
        d1 = self.dec_res1(d1, t_emb) # [B, 64, 14, 14]

        d2 = self.up2(d1) # [B, 32, 28, 28]
        # Skip connection from h1
        d2 = torch.cat([d2, h1], dim=1) # [B, 32+32, 28, 28]
        d2 = self.dec_res2(d2, t_emb) # [B, 32, 28, 28]

        # Output
        output = self.out(d2) # [B, 1, 28, 28]
        return output

# === improved Gaussian Diffusion with clip ===
# (Unchanged from v7)
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
        x0_pred = x0_pred.clamp(-1, 1) # Clip predicted x0

        if t[0] > 0:
            alpha_bar_prev = self.alpha_bar[t - 1].view(-1, 1, 1, 1)
            # DDPM sampling formula components
            posterior_mean = (beta_t * alpha_bar_prev.sqrt() / (1 - alpha_bar_t)) * x0_pred + \
                             ((1 - alpha_bar_prev) * alpha_t.sqrt() / (1 - alpha_bar_t)) * x
            posterior_variance = (beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t))
            posterior_log_variance = torch.log(posterior_variance.clamp(min=1e-20)) # Avoid log(0)
            std = torch.exp(0.5 * posterior_log_variance)
            noise = torch.randn_like(x)
            return posterior_mean + std * noise
        else:
            # At t=0, the output is the clipped predicted x0
            return x0_pred

# === Sampling ===
# (Unchanged from v7)
@torch.no_grad()
def sample(model, diffusion, steps=1000, batch_size=1): # Added batch_size option
    model.eval()
    x = torch.randn(batch_size, 1, 28, 28, device=DEVICE)
    for t in reversed(range(steps)):
        t_tensor = torch.tensor([t] * batch_size, device=DEVICE).long() # Create t_tensor for batch
        x = diffusion.p_sample(model, x, t_tensor)
    return x

# === Training ===
# (Mostly unchanged, ensures correct model and diffusion instances are used)
def train_pipeline(digit_label=1, use_quantum=False, arm=None,
                   save_dir_base="diffusion_models_v8", epochs=30, batch_size=64,
                   lr=3e-4, n_train=None, seed=0, n_hidden=16, max_steps=None,
                   sample_every=0, device_name="default.qubit",
                   diff_method="backprop"):
    """Train one arm on one digit.

    Args:
        arm: one of bottlenecks.ARMS; overrides `use_quantum`. Note "se_frozen"
            is not available on this model -- its circuit pools its own input, so
            there is no down-projection to freeze.
        n_train: training images (issue #5's x-axis). None = the whole class.
        seed: seeds init, noise *and* the data subset.
        max_steps: budget in optimizer steps. Required for any comparison across
            n_train: an epoch at N=10 is a single gradient step, so an
            epoch-based budget leaves the low-N arms undertrained rather than
            data-limited. Falls back to epochs * steps_per_epoch when omitted.
        sample_every: also dump samples every N steps (0 = only at the end).
    """
    if arm is None:
        arm = "quantum" if use_quantum else "plain"
    torch.manual_seed(seed)

    run_name = (f"mnist_{digit_label}_{arm}_n{n_train if n_train is not None else 'all'}"
                f"_s{seed}_t{max_steps if max_steps is not None else f'e{epochs}'}")
    save_dir = os.path.join(save_dir_base, run_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving results to: {save_dir}")

    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    # Seeded subset of this digit (shared with v7 via sweep_utils).
    indices, n_used = seeded_subset_indices(dataset.targets, digit_label, n_train, seed)
    subset_dataset = torch.utils.data.Subset(dataset, indices.tolist())

    loader = DataLoader(subset_dataset, batch_size=min(batch_size, n_used),
                        shuffle=True, num_workers=4, pin_memory=True)

    quantum_kwargs = ({"device_name": device_name, "diff_method": diff_method}
                      if arm == "quantum" else {})
    model = ImprovedUNet(arm=arm, n_hidden=n_hidden, **quantum_kwargs).to(DEVICE)
    bn_trainable, bn_total = model.bottleneck_parameter_counts()
    manifest = {
        "script": "v8", "arm": arm, "digit": digit_label, "n_train": n_used,
        "seed": seed, "epochs": epochs, "batch_size": loader.batch_size, "lr": lr,
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
    ema_model = ImprovedUNet(arm=arm, n_hidden=n_hidden, **quantum_kwargs).to(DEVICE)
    ema_model.load_state_dict(model.state_dict())
    diffusion = ImprovedGaussianDiffusion()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
    model.train()
    bar = tqdm(range(1, total_steps + 1), desc=run_name)
    for step in bar:
        x, _ = next(stream)
        x = x.to(DEVICE)
        optimizer.zero_grad()
        t = torch.randint(0, diffusion.timesteps, (x.size(0),), device=DEVICE).long()
        noise = torch.randn_like(x)
        x_noisy = diffusion.q_sample(x, t, noise)
        noise_pred = model(x_noisy, t)
        loss = F.mse_loss(noise_pred, noise)

        loss.backward()
        optimizer.step()

        # EMA Update
        with torch.no_grad():
            ema_decay = 0.999
            for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                p_ema.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

        window.append(loss.item())
        # Checkpoint on the mean over one epoch-equivalent of steps, so the
        # selection criterion does not change meaning when the budget is given in
        # steps rather than epochs.
        if step % steps_per_epoch == 0 or step == total_steps:
            mean_loss = sum(window) / len(window)
            loss_history.append((step, mean_loss))
            bar.set_postfix(mean_loss=f"{mean_loss:.4f}")
            window = []
            if mean_loss < best_loss:
                best_loss = mean_loss
                torch.save(ema_model.state_dict(), os.path.join(save_dir, "best_model.pth"))

        if sample_every and step % sample_every == 0:
            dump_samples(ema_model, diffusion, save_dir, step, digit_label)

    dump_samples(ema_model, diffusion, save_dir, total_steps, digit_label)

    # Save final model
    torch.save(ema_model.state_dict(), os.path.join(save_dir, "final_model.pth"))

    # Save loss history
    with open(os.path.join(save_dir, "loss.csv"), "w") as f:
        f.write("step,mean_loss\n")
        for step_i, l in loss_history:
            f.write(f"{step_i},{l}\n")

    # Plot loss
    plt.figure()
    plt.plot([st for st, _ in loss_history], [l for _, l in loss_history])
    plt.title(f"Training Loss (Label {digit_label}, arm={arm})")
    plt.xlabel("Step")
    plt.ylabel("Mean MSE Loss (window)")
    plt.grid(True)
    plt.savefig(f"{save_dir}/loss_curve.png")
    plt.close()

def dump_samples(ema_model, diffusion, save_dir, step, digit_label, n=5):
    """Save the sampler's raw output (.pt) plus a contact sheet for eyeballing.

    The .pt is what evaluate.py reads. Do NOT compute metrics from the .png:
    imshow rescales each subplot to its own min/max and the figure is rasterized
    at the figure DPI (issue #4).
    """
    was_training = ema_model.training
    ema_model.eval()
    sample_images = sample(ema_model, diffusion, steps=diffusion.timesteps,
                           batch_size=n)[0:n]
    torch.save(sample_images.cpu(), f"{save_dir}/step{step:06d}_samples.pt")
    plt.figure(figsize=(10, 2))
    for i in range(sample_images.size(0)):
        plt.subplot(1, n, i + 1)
        plt.imshow(sample_images[i].squeeze().cpu().numpy(), cmap='gray')
        plt.axis("off")
    plt.suptitle(f"Step {step} Samples (Label {digit_label})")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{save_dir}/step{step:06d}_samples.png")
    plt.close()
    if was_training:
        ema_model.train()


def main():
    ap = argparse.ArgumentParser(
        description="Train the v8 full-U-Net MNIST diffusion model, one arm of "
                    "the #3 ablation ladder at a time.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""note:
  This model's QuantumLayer pools its own input down to n_qubits, so it has no
  trainable down-projection and the 'se_frozen' arm does not apply here. The
  classical control ('se') pools too, so that 'se' and 'quantum' stay matched to
  within the circuit's own parameters.

examples:
  python quantum_diffusion_mnist_v8.py --arm quantum --digits 3 --n-train 100
  python quantum_diffusion_mnist_v8.py --arm se      --digits 3 --n-train 100
""")
    ap.add_argument("--arm", choices=[a for a in ARMS if a != "se_frozen"],
                    default="quantum",
                    help="bottleneck arm. 'se' is the parameter-matched classical "
                         "control; 'plain' has no gating module at all.")
    ap.add_argument("--digits", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--n-train", type=int, default=None,
                    help="training images per class; omit for the whole class")
    ap.add_argument("--seed", type=int, default=0,
                    help="seeds init, noise and the data subset")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="budget in optimizer steps. Strongly preferred over "
                         "--epochs for any comparison across --n-train, since an "
                         "epoch at N=10 is a single gradient step.")
    ap.add_argument("--epochs", type=int, default=30,
                    help="used only when --max-steps is omitted")
    ap.add_argument("--sample-every", type=int, default=0,
                    help="also dump samples every N steps (0 = only at the end)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-hidden", type=int, default=16,
                    help="bottleneck width; the quantum arm's qubit count")
    ap.add_argument("--qdevice", default="default.qubit",
                    help="lightning.qubit with --diff-method adjoint is ~3.5x "
                         "faster and numerically equivalent")
    ap.add_argument("--diff-method", default="backprop",
                    choices=["backprop", "adjoint", "parameter-shift"])
    ap.add_argument("--save-dir", default=None, help="default: diffusion_models_v8_<arm>")
    args = ap.parse_args()

    save_dir_base = args.save_dir or f"diffusion_models_v8_{args.arm}"
    for digit in args.digits:
        print(f"\n--- v8 arm={args.arm} digit={digit} ---")
        train_pipeline(digit_label=digit, arm=args.arm, save_dir_base=save_dir_base,
                       epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                       n_train=args.n_train, seed=args.seed, n_hidden=args.n_hidden,
                       max_steps=args.max_steps, sample_every=args.sample_every,
                       device_name=args.qdevice, diff_method=args.diff_method)


if __name__ == '__main__':
    main()
