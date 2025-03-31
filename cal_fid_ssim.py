import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as ssim
from torchvision.datasets import MNIST
from torchvision.models import inception_v3, Inception_V3_Weights
import torch
from scipy import linalg
from tqdm import tqdm

# === Step 1: Auto-cut based on column gaps ===
def split_images_by_column_gap(image_path, output_size=(28, 28), threshold=200, save_dir=None):
    img = Image.open(image_path).convert('L')
    img_np = np.array(img)

    # Remove border white space
    binary = (img_np < threshold).astype(np.uint8)
    row_sum = binary.sum(axis=1)
    col_sum = binary.sum(axis=0)
    row_idxs = np.where(row_sum > 0)[0]
    col_idxs = np.where(col_sum > 0)[0]

    if len(row_idxs) == 0 or len(col_idxs) == 0:
        print("Could not detect digit area.")
        return []

    y1, y2 = row_idxs.min(), row_idxs.max()
    x1, x2 = col_idxs.min(), col_idxs.max()
    cropped = img_np[y1:y2+1, x1:x2+1]
    cropped_binary = binary[y1:y2+1, x1:x2+1]

    # Analyze vertical whitespace
    col_activity = cropped_binary.sum(axis=0)
    digit_regions = []
    in_digit = False
    start = 0

    for i, val in enumerate(col_activity):
        if val > 0 and not in_digit:
            in_digit = True
            start = i
        elif val == 0 and in_digit:
            in_digit = False
            end = i
            digit_regions.append((start, end))

    if in_digit:
        digit_regions.append((start, len(col_activity)))

    images = []
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for i, (x_start, x_end) in enumerate(digit_regions):
        digit_crop = cropped[:, x_start:x_end]
        digit_img = Image.fromarray(digit_crop).resize(output_size, Image.BICUBIC)
        digit_np = np.array(digit_img)
        images.append(digit_np)

        if save_dir:
            save_path = os.path.join(save_dir, f"digit_{i:03d}.png")
            digit_img.save(save_path)

    print(f"Detected {len(images)} digits.")
    return np.array(images)

# === Step 2: Load real MNIST ===
def load_real_mnist(n, transform=None):
    dataset = MNIST(root='./data', train=True, download=True)
    if transform:
        images = [transform(dataset[i][0]) for i in range(n)]
    else:
        images = [np.array(dataset[i][0]) for i in range(n)]
    return np.array(images)

# === Step 3: SSIM ===
def calculate_ssim(generated, real):
    ssim_scores = []
    for i in range(min(len(generated), len(real))):
        score = ssim(generated[i], real[i], data_range=255)
        ssim_scores.append(score)
    return np.mean(ssim_scores)

# === Step 4: FID ===
def get_inception_activations(images, model, batch_size=32, device='cpu'):
    model.eval()
    activations = []
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # grayscale -> 3-channel
    ])
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = [transform(img.astype(np.uint8)) for img in images[i:i + batch_size]]
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_path = '/home/meow/quantum_diffusion/improved_diffusion_quantum/mnist_9/epoch30_samples.png'
    output_dir = os.path.dirname(input_path)
    split_output_dir = os.path.join(output_dir, 'split_digits')

    print("Splitting stitched image...")
    generated_imgs = split_images_by_column_gap(input_path, save_dir=split_output_dir)

    if len(generated_imgs) == 0:
        print("No digits extracted. Try adjusting the threshold or input image.")
        return

    print(f"Loaded {len(generated_imgs)} generated digits.")
    real_imgs = load_real_mnist(len(generated_imgs))

    print("Calculating SSIM...")
    avg_ssim = calculate_ssim(generated_imgs, real_imgs)
    print(f"Average SSIM: {avg_ssim:.4f}")

    print("Loading InceptionV3 for FID...")
    weights = Inception_V3_Weights.DEFAULT
    inception = inception_v3(weights=weights, aux_logits=True)
    inception.fc = torch.nn.Identity()
    inception.to(device)

    print("Calculating FID...")
    act_gen = get_inception_activations(generated_imgs, inception, device=device)
    act_real = get_inception_activations(real_imgs, inception, device=device)
    fid_score = calculate_fid(act_gen, act_real)
    print(f"FID Score: {fid_score:.2f}")

    output_path = os.path.join(output_dir, 'fid_ssim.txt')
    with open(output_path, 'w') as f:
        f.write(f"FID Score: {fid_score:.2f}\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")

    print(f"Results saved to: {output_path}")
    print(f"Split digits saved to: {split_output_dir}")

if __name__ == '__main__':
    main()
