# === Quantum Diffusion MNIST v8 (Improved UNet, ResBlocks, Time Embedding) ===

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
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
    def __init__(self, n_qubits=16, n_layers=3, embed_dim=TIME_EMBED_DIM):
        super().__init__()
        self.n_qubits = n_qubits
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            # Ensure input features match n_qubits
            inputs_resized = F.adaptive_avg_pool1d(inputs.unsqueeze(1), n_qubits).squeeze(1)
            for i in range(n_qubits):
                # Scale inputs to typical angle range like [0, pi]
                qml.RY(torch.pi * torch.sigmoid(inputs_resized[:, i]), wires=i)
            for l in range(n_layers):
                for i in range(n_qubits):
                    qml.RZ(weights[l, i], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                # Optional: Entangling layer like CNOT ladder or ring
                # qml.CNOT(wires=[n_qubits - 1, 0]) # Example: cycle entanglement
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        # Projects final qubit measurements back to the embedding dimension
        self.output_proj = nn.Linear(n_qubits, embed_dim)

    def forward(self, x):
        # x shape: [Batch, embed_dim]
        # No input projection needed if embed_dim is handled correctly upstream
        x_cpu = x.detach().cpu() # QNode execution on CPU
        quantum_output = self.qlayer(x_cpu) # Shape: [Batch, n_qubits]
        quantum_output = quantum_output.to(x.device) # Move back to original device
        return self.output_proj(quantum_output) # Project back to embed_dim


# === Improved UNet with ResBlocks, Time Embedding, and Skip Connections ===
class ImprovedUNet(nn.Module):
    def __init__(self, use_quantum=False, time_embed_dim=TIME_EMBED_DIM):
        super().__init__()
        self.use_quantum = use_quantum
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
        if use_quantum:
            # Ensure quantum layer input dim matches the bottleneck feature dim if applied differently
            # Here, applying it to the pooled features, so input is 128
            self.q_attn = QuantumLayer(embed_dim=128) # Assuming bottleneck dim is 128

        # Decoder
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2) # Upsample
        # Decoder ResBlock takes concatenated input (skip + upsampled)
        self.dec_res1 = ResBlock(64 + 64, 64, time_embed_dim) # Corrected input channels
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2) # Upsample
        self.dec_res2 = ResBlock(32 + 32, 32, time_embed_dim) # Corrected input channels

        # Final Layer
        self.out = nn.Conv2d(32, 1, 1) # Use 1x1 conv for final projection

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

        if self.use_quantum:
             # Apply quantum attention similar to v7, on pooled features
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
def train_pipeline(digit_label=1, use_quantum=False, save_dir_base="diffusion_models_v8", epochs=30, batch_size=64, lr=3e-4):
    save_dir = os.path.join(save_dir_base, f"mnist_{'quantum' if use_quantum else 'classical'}", f"label_{digit_label}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving results to: {save_dir}")

    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    # Filter for the specific digit
    indices = [i for i, target in enumerate(dataset.targets) if target == digit_label]
    subset_dataset = torch.utils.data.Subset(dataset, indices)

    if not subset_dataset:
         print(f"Warning: No data found for label {digit_label}. Skipping training.")
         return

    loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = ImprovedUNet(use_quantum=use_quantum).to(DEVICE)
    ema_model = copy.deepcopy(model)
    diffusion = ImprovedGaussianDiffusion()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    best_loss = float('inf')

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        model.train() # Set model to training mode
        for x, _ in tqdm(loader, desc=f"Label {digit_label} - Epoch {epoch+1}/{epochs}"):
            x = x.to(DEVICE)
            optimizer.zero_grad()
            t = torch.randint(0, diffusion.timesteps, (x.size(0),), device=DEVICE).long()
            noise = torch.randn_like(x)
            x_noisy = diffusion.q_sample(x, t, noise)
            noise_pred = model(x_noisy, t)
            loss = F.mse_loss(noise_pred, noise)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            # EMA Update
            with torch.no_grad():
                ema_decay = 0.999
                for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                    p_ema.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        loss_history.append(avg_epoch_loss)
        print(f"[Label {digit_label} Epoch {epoch+1}] Avg Loss: {avg_epoch_loss:.4f}")

        # Save sample images using EMA model
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1: # Sample every 5 epochs and at the end
             ema_model.eval() # Set EMA model to eval mode for sampling
             sample_images = sample(ema_model, diffusion, steps=diffusion.timesteps, batch_size=5)[0:5] # Generate 5 samples
             plt.figure(figsize=(10, 2))
             for i in range(sample_images.size(0)):
                 img = sample_images[i].squeeze().cpu().numpy()
                 plt.subplot(1, 5, i + 1)
                 plt.imshow(img, cmap='gray')
                 plt.axis("off")
             plt.suptitle(f"Epoch {epoch+1} Samples (Label {digit_label})")
             plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
             plt.savefig(f"{save_dir}/epoch{epoch+1:03d}_samples.png")
             plt.close()

        # Save best model based on average epoch loss
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(ema_model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"Saved new best model with loss {best_loss:.4f}")

    # Save final model
    torch.save(ema_model.state_dict(), os.path.join(save_dir, "final_model.pth"))

    # Save loss history to txt
    with open(os.path.join(save_dir, "loss.txt"), "w") as f:
        for l in loss_history:
            f.write(f"{l}\n")

    # Plot loss
    plt.figure()
    plt.plot(loss_history)
    plt.title(f"Training Loss (Label {digit_label}, {'Quantum' if use_quantum else 'Classical'})")
    plt.xlabel("Epoch")
    plt.ylabel("Average MSE Loss")
    plt.grid(True)
    plt.savefig(f"{save_dir}/loss_curve.png")
    plt.close()

if __name__ == '__main__':
    # Example: Train classical models for digits 0 and 1, and quantum for digit 2
    # Classical models
    # train_pipeline(digit_label=0, use_quantum=False, save_dir_base="diffusion_models_v8_classical", epochs=50)
    # train_pipeline(digit_label=1, use_quantum=False, save_dir_base="diffusion_models_v8_classical", epochs=50)

    # Quantum model
    # train_pipeline(digit_label=2, use_quantum=True, save_dir_base="diffusion_models_v8_quantum", epochs=50)

    # Train all quantum models (example)
    # for digit in range(10): # Train for digits 0 through 9
    #      print(f"\n--- Training Quantum Model for Digit: {digit} ---")
    #      train_pipeline(
    #          digit_label=digit,
    #          use_quantum=True,
    #          save_dir_base="diffusion_models_v8_quantum_all_digits",
    #          epochs=30, # Adjust epochs as needed
    #          batch_size=64,
    #          lr=3e-4
    #      )
         # Optional: Clear CUDA cache between runs if memory issues arise
         # if torch.cuda.is_available():
         #     torch.cuda.empty_cache()

    # Train all classical models (example)
    for digit in range(10):
        print(f"\n--- Training Classical Model for Digit: {digit} ---")
        train_pipeline(
            digit_label=digit,
            use_quantum=False,
            save_dir_base="diffusion_models_v8_classical_all_digits",
            epochs=30,
            batch_size=64,
            lr=3e-4
        )
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

    print("\n=== Training Complete ===") 