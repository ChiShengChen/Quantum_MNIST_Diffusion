import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim
from torchvision.datasets import MNIST
from torchvision.models import inception_v3, Inception_V3_Weights
import torch
from scipy import linalg
from tqdm import tqdm
import csv

# === Device Setup ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# === Functions from cal_fid_ssim.py (slightly modified for clarity/robustness) ===

def split_images_by_column_gap(image_path, output_size=(28, 28), threshold=200, min_width=5, save_dir=None):
    """Splits a horizontally stitched image into individual digits based on vertical gaps."""
    try:
        img = Image.open(image_path).convert('L')
        img_np = np.array(img)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return np.array([])
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return np.array([])

    # Find bounding box of all digits
    binary = (img_np < threshold).astype(np.uint8)
    row_sum = binary.sum(axis=1)
    col_sum = binary.sum(axis=0)
    row_idxs = np.where(row_sum > 0)[0]
    col_idxs = np.where(col_sum > 0)[0]

    if len(row_idxs) == 0 or len(col_idxs) == 0:
        print(f"Warning: No significant content detected in {image_path}.")
        return np.array([])

    y1, y2 = row_idxs.min(), row_idxs.max()
    x1, x2 = col_idxs.min(), col_idxs.max()
    cropped = img_np[y1:y2+1, x1:x2+1]
    cropped_binary = binary[y1:y2+1, x1:x2+1]

    # Detect digit regions based on vertical gaps
    col_activity = cropped_binary.sum(axis=0)
    digit_regions = []
    in_digit = False
    start = 0
    for i, val in enumerate(col_activity):
        is_active = val > 0
        if is_active and not in_digit:
            in_digit = True
            start = i
        elif not is_active and in_digit:
            in_digit = False
            end = i
            if end - start >= min_width: # Only consider regions wide enough
                 digit_regions.append((start, end))
        elif i == len(col_activity) - 1 and in_digit: # Handle digit reaching the edge
            end = i + 1
            if end - start >= min_width:
                digit_regions.append((start, end))


    images = []
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Extract, resize, and optionally save each digit
    for i, (x_start, x_end) in enumerate(digit_regions):
        digit_crop = cropped[:, x_start:x_end]
        try:
            digit_img = Image.fromarray(digit_crop).resize(output_size, Image.BICUBIC)
            digit_np = np.array(digit_img)
            images.append(digit_np)

            if save_dir:
                save_path = os.path.join(save_dir, f"split_digit_{i:03d}.png")
                digit_img.save(save_path)
        except Exception as e:
             print(f"Error processing split digit {i} from {image_path}: {e}")


    print(f"Detected and extracted {len(images)} digits from {os.path.basename(image_path)}.")
    return np.array(images)


def load_real_mnist_by_label(n, label, root='./data'):
    """Loads n real MNIST images for a specific label."""
    dataset = MNIST(root=root, train=True, download=True)
    indices = [i for i, target in enumerate(dataset.targets) if target == label]
    if len(indices) < n:
        print(f"Warning: Only found {len(indices)} real images for label {label}, requested {n}.")
        n = len(indices)

    selected_indices = np.random.choice(indices, n, replace=False)
    images = [np.array(dataset[i][0]) for i in selected_indices]
    return np.array(images)


def calculate_ssim(generated, real):
    """Calculates average SSIM between two sets of grayscale images."""
    if generated.shape[0] == 0 or real.shape[0] == 0:
        return 0.0
    num_compare = min(len(generated), len(real))
    ssim_scores = []
    for i in range(num_compare):
        # Ensure images are 2D grayscale
        gen_img = generated[i].squeeze()
        real_img = real[i].squeeze()
        if gen_img.ndim != 2 or real_img.ndim != 2:
             print(f"Warning: Skipping SSIM for image {i} due to unexpected dimensions.")
             continue
        try:
            score = ssim(gen_img, real_img, data_range=gen_img.max() - gen_img.min())
            ssim_scores.append(score)
        except ValueError as e:
             print(f"Warning: SSIM calculation failed for image {i}. Error: {e}")

    return np.mean(ssim_scores) if ssim_scores else 0.0


def get_inception_activations(images, model, batch_size=32, device=DEVICE):
    """Calculates InceptionV3 activations for a list of images."""
    model.eval()
    activations = []
    # Pre-define transform to handle grayscale to RGB conversion needed by InceptionV3
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        transforms.Grayscale(num_output_channels=3), # Convert grayscale to 3 channels
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalization expected by InceptionV3
    ])

    dataloader = torch.utils.data.DataLoader(
        [(transform(img.astype(np.uint8))) for img in images],
        batch_size=batch_size
    )

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating Activations", leave=False):
             batch = batch.to(device)
             # Get features from the final average pooling layer
             pred = model(batch)
             # Handle cases where aux_logits are returned (InceptionOutputs object)
             if isinstance(pred, torch.Tensor):
                 activations.append(pred.cpu().numpy())
             elif hasattr(pred, 'logits'): # Newer torchvision returns InceptionOutputs
                 activations.append(pred.logits.cpu().numpy())
             else: # Fallback for older versions or different return types
                 activations.append(pred[0].cpu().numpy())


    return np.concatenate(activations, axis=0)


def calculate_fid(act1, act2):
    """Calculates the Frechet Inception Distance (FID) between two sets of activations."""
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    # Add small epsilon for numerical stability if covariance matrix is singular
    epsilon = 1e-6
    sigma1 += np.eye(sigma1.shape[0]) * epsilon
    sigma2 += np.eye(sigma2.shape[0]) * epsilon


    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # Calculate sqrtm using scipy.linalg
    try:
        covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    except Exception as e:
        print(f"Error calculating sqrtm: {e}. FID calculation may be inaccurate.")
        return np.inf # Return infinity or a very large number on error


    # Check for complex numbers and take real part if necessary
    if np.iscomplexobj(covmean):
        # Check if imaginary part is negligible
        if np.max(np.abs(covmean.imag)) > 1e-5:
             print(f"Warning: Significant imaginary part in sqrtm ({np.max(np.abs(covmean.imag))}). Taking real part.")
        covmean = covmean.real

    # FID calculation
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


# === Main Loop ===
def evaluate_all_digits(base_dir, output_csv, sample_filename="epoch030_samples.png", num_real_samples=1000):
    """Calculates FID and SSIM for each digit subdirectory and saves to CSV."""

    print("Loading InceptionV3 model...")
    weights = Inception_V3_Weights.DEFAULT
    inception_model = inception_v3(weights=weights, aux_logits=True) # Corrected: Must be True for default weights
    # Modify the final layer to get features instead of classifications
    inception_model.fc = torch.nn.Identity()
    inception_model.to(DEVICE)
    inception_model.eval() # Ensure model is in eval mode

    results = []

    print(f"Processing directories in: {base_dir}")
    for digit in range(10):
        print(f"\n--- Processing Digit: {digit} ---")
        digit_dir = os.path.join(base_dir, f"label_{digit}")
        sample_image_path = os.path.join(digit_dir, sample_filename)
        split_save_dir = os.path.join(digit_dir, "split_digits") # Optional: save split images

        if not os.path.exists(digit_dir):
            print(f"Directory not found: {digit_dir}. Skipping.")
            results.append({'Digit': digit, 'FID': 'N/A', 'SSIM': 'N/A', 'Generated Count': 0, 'Status': 'Directory Missing'})
            continue

        if not os.path.exists(sample_image_path):
            print(f"Sample image not found: {sample_image_path}. Skipping FID/SSIM.")
            results.append({'Digit': digit, 'FID': 'N/A', 'SSIM': 'N/A', 'Generated Count': 0, 'Status': 'Sample Missing'})
            continue

        print(f"Splitting image: {sample_image_path}")
        # Use save_dir=split_save_dir if you want to save the individual digits
        generated_imgs = split_images_by_column_gap(sample_image_path, save_dir=None)

        if len(generated_imgs) == 0:
            print("No digits extracted from the sample image.")
            results.append({'Digit': digit, 'FID': 'N/A', 'SSIM': 'N/A', 'Generated Count': 0, 'Status': 'Split Failed'})
            continue

        num_generated = len(generated_imgs)
        print(f"Loading {num_real_samples} real MNIST images for digit {digit}...")
        # Ensure we load enough real images for a stable FID, but also compare SSIM fairly
        real_imgs = load_real_mnist_by_label(max(num_generated, num_real_samples), digit)
        num_real_loaded = len(real_imgs)

        # --- Calculate SSIM ---
        print("Calculating SSIM...")
        # Use the first num_generated real images for a fair comparison
        avg_ssim = calculate_ssim(generated_imgs, real_imgs[:num_generated])
        print(f"Average SSIM: {avg_ssim:.4f}")

        # --- Calculate FID ---
        print("Calculating FID...")
        # Use all loaded real images for FID calculation
        act_gen = get_inception_activations(generated_imgs, inception_model, device=DEVICE)
        act_real = get_inception_activations(real_imgs, inception_model, device=DEVICE)
        fid_score = calculate_fid(act_gen, act_real)
        print(f"FID Score: {fid_score:.2f}")

        results.append({
            'Digit': digit,
            'FID': f"{fid_score:.2f}",
            'SSIM': f"{avg_ssim:.4f}",
            'Generated Count': num_generated,
            'Real Count Used (FID)': num_real_loaded,
            'Status': 'Success'
        })


    # --- Write results to CSV ---
    print(f"\nWriting results to {output_csv}...")
    if not results:
        print("No results generated.")
        return

    fieldnames = ['Digit', 'FID', 'SSIM', 'Generated Count', 'Real Count Used (FID)', 'Status']
    try:
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print("CSV file saved successfully.")
    except IOError as e:
        print(f"Error writing CSV file: {e}")


if __name__ == '__main__':
    # Configure the base directory where label_0, label_1, etc. folders are located
    classical_base_dir = "diffusion_models_v8_classical_all_digits/mnist_classical"
    quantum_base_dir = "diffusion_models_v8_quantum_all_digits/mnist_quantum"
    # Output CSV file name
    output_csv_file = os.path.join(os.path.dirname(classical_base_dir), "classical_metrics_summary.csv")
    output_csv_file_quantum = os.path.join(os.path.dirname(quantum_base_dir), "quantum_metrics_summary.csv")
    # Name of the sample image file generated by the training script
    sample_img_name = "epoch030_samples.png" # Adjust if your filename is different (e.g., final_samples.png)
    # Number of real images to load for FID comparison (more is generally better/more stable)
    num_real_for_fid = 5000 # Using 5k real images for FID

    evaluate_all_digits(classical_base_dir, output_csv_file, sample_img_name, num_real_for_fid)
    evaluate_all_digits(quantum_base_dir, output_csv_file_quantum, sample_img_name, num_real_for_fid)
    print("\nEvaluation complete.") 