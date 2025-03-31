import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from medmnist import PathMNIST
from scipy import linalg

# === Load fixed generated images ===
def load_fixed_generated_images(base_path, indices, size=(28, 28)):
    images = []
    for i in indices:
        path = os.path.join(base_path, f"epoch30_sample{i}.png")
        img = Image.open(path).convert("RGB").resize(size)
        images.append(np.array(img))
    return np.array(images)

# === Load real PathMNIST label=1 ===
def load_real_pathmnist(n):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    dataset = PathMNIST(split='train', download=True, root='./data')
    real_imgs = []
    for img, label in dataset:
        if label == 1:
            tensor_img = transform(img).permute(1, 2, 0).numpy() * 255
            real_imgs.append(tensor_img.astype(np.uint8))
        if len(real_imgs) >= n:
            break
    return np.array(real_imgs)


# === SSIM ===
def calculate_ssim(generated, real):
    ssim_scores = []
    for i in range(min(len(generated), len(real))):
        score = ssim(generated[i], real[i], channel_axis=2, data_range=255)
        ssim_scores.append(score)
    return np.mean(ssim_scores)

# === FID ===
def get_inception_activations(images, model, batch_size=32, device='cpu'):
    model.eval()
    activations = []
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        transforms.ToTensor()
    ])
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = [transform(img) for img in images[i:i + batch_size]]
            batch = torch.stack(batch).to(device)
            pred = model(batch)
            activations.append(pred.cpu().numpy())
    return np.concatenate(activations, axis=0)

def calculate_fid(act1, act2):
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)

# === Main ===
def main():
    generated_dir = "/home/meow/quantum_diffusion/diffusion_pathmnist_quantum_label1"
    output_txt = os.path.join(generated_dir, "fid_ssim_pathmnist.txt")
    indices = list(range(5))  # epoch30_sample0 ~ sample4

    print("Loading generated images...")
    generated_imgs = load_fixed_generated_images(generated_dir, indices)
    print(f"Loaded {len(generated_imgs)} generated images.")

    print("Loading real PathMNIST label=1...")
    real_imgs = load_real_pathmnist(len(generated_imgs))
    print(f"Loaded {len(real_imgs)} real images.")

    print("Calculating SSIM...")
    avg_ssim = calculate_ssim(generated_imgs, real_imgs)
    print(f"SSIM: {avg_ssim:.4f}")

    print("Calculating FID...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights, aux_logits=True)
    model.fc = torch.nn.Identity()
    model.to(device)

    act_gen = get_inception_activations(generated_imgs, model, device=device)
    act_real = get_inception_activations(real_imgs, model, device=device)
    fid_score = calculate_fid(act_gen, act_real)
    print(f"FID: {fid_score:.2f}")

    with open(output_txt, "w") as f:
        f.write(f"FID Score: {fid_score:.2f}\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")

    print(f"Results saved to: {output_txt}")

if __name__ == '__main__':
    main()
