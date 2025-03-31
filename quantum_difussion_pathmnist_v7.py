import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
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
    def __init__(self, n_qubits=16, n_layers=3):
        super().__init__()
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            for l in range(n_layers):
                for i in range(n_qubits):
                    qml.RZ(weights[l, i], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        self.input_proj = nn.Linear(128, n_qubits)
        self.output_proj = nn.Linear(n_qubits, 128)

    def forward(self, x):
        x = self.input_proj(x)
        x_cpu = x.detach().cpu()
        results = [self.qlayer(sample) for sample in x_cpu]
        out = torch.stack(results).to(x.device)
        return self.output_proj(out)


# === UNet ===
class ImprovedUNet(nn.Module):
    def __init__(self, channels=3, use_quantum=False):
        super().__init__()
        self.use_quantum = use_quantum
        self.enc1 = nn.Conv2d(channels, 32, 3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.res_block = ResBlock(128)
        self.time_embed_proj = nn.Linear(128, 128)
        if use_quantum:
            self.q_attn = QuantumLayer()
        self.dec1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.out = nn.Conv2d(64, channels, 3, padding=1)

    def forward(self, x, t):
        t_embed = get_timestep_embedding(t, 128).to(x.device)
        t_proj = self.time_embed_proj(t_embed).unsqueeze(-1).unsqueeze(-1)
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(e1))
        e3 = F.relu(self.enc3(e2)) + t_proj
        x = self.res_block(e3)
        pooled = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
        if self.use_quantum:
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
def sample(model, diffusion, steps=1000):
    model.eval()
    x = torch.randn(1, 3, 28, 28, device=DEVICE)
    for t in reversed(range(steps)):
        t_tensor = torch.tensor([t], device=DEVICE).long()
        x = diffusion.p_sample(model, x, t_tensor)
    return x


# === Dataset ===
def load_pathmnist(label_condition=1):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])
    dataset = PathMNIST(split='train', download=True, transform=transform)

    if label_condition is not None:
        # setting label
        indices = [i for i, (_, label) in enumerate(dataset) if label == label_condition]
        dataset = torch.utils.data.Subset(dataset, indices)

    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    return loader



# === Training ===
def train_pipeline(use_quantum=False, save_dir="diffusion_pathmnist_classical", label_condition=1):
    if label_condition is not None:
        save_dir += f"_label{label_condition}"
    os.makedirs(save_dir, exist_ok=True)

    loader = load_pathmnist(label_condition=label_condition)
    model = ImprovedUNet(channels=3, use_quantum=use_quantum).to(DEVICE)
    ema_model = copy.deepcopy(model)
    diffusion = ImprovedGaussianDiffusion()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_history = []
    best_loss = float('inf')

    for epoch in range(30):
        for x, _ in tqdm(loader, desc=f"Epoch {epoch+1}"):
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

        loss_history.append(loss.item())
        print(f"[Epoch {epoch+1}] Loss: {loss.item():.4f}")

        # Save samples (5 images)
        samples = [sample(ema_model, diffusion)[0].cpu() for _ in range(5)]
        grid = vutils.make_grid(samples, nrow=5, normalize=True, value_range=(-1, 1))
        plt.figure(figsize=(10, 2))
        plt.imshow(grid.permute(1, 2, 0).numpy())
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/epoch{epoch+1}_samples.png")
        plt.close()

        # Save single images
        for i, img in enumerate(samples):
            img_np = (img.permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)
            plt.imsave(f"{save_dir}/epoch{epoch+1}_sample{i}.png", img_np)

        # Save best model
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(ema_model.state_dict(), os.path.join(save_dir, "best_model.pth"))

    # Save loss
    with open(os.path.join(save_dir, "loss.txt"), "w") as f:
        for l in loss_history:
            f.write(f"{l}\n")

    plt.plot(loss_history)
    plt.title("Training Loss (PathMNIST)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.savefig(f"{save_dir}/loss_curve.png")
    plt.close()


# === Run ===
if __name__ == '__main__':
    # train_pipeline(use_quantum=True)
    train_pipeline(use_quantum=False, save_dir="diffusion_pathmnist_classical")
    train_pipeline(use_quantum=True, save_dir="diffusion_pathmnist_quantum")
