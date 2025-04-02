# Quantum Diffusion Models
[![arXiv](https://img.shields.io/badge/arXiv-2504.00034-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2504.00034)  
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.30%2B-green)](https://pennylane.ai/)

The official implement of Quantum Generative Models for Image Generation: Insights from MNIST and MedMNIST. A novel approach to image generation using quantum-enhanced diffusion models. This project implements diffusion models enhanced with quantum circuits for medical and standard image generation.

## 📋 Overview

<img width="459" alt="image" src="https://github.com/user-attachments/assets/d89c02af-5a27-4759-9734-80deca1a1836" />
<img width="450" alt="image" src="https://github.com/user-attachments/assets/7c0a4fda-27c6-4dae-97c8-7a09c3d1166b" />


This repository explores the integration of quantum computing into diffusion models for image generation. The implementation provides both classical and quantum-enhanced versions of diffusion models for MNIST and PathMNIST datasets.

Key features:
- Quantum-enhanced attention mechanism for diffusion models
- Classical vs quantum model comparison framework
- Evaluation metrics (FID, SSIM) for generated images
- Support for MNIST and PathMNIST medical datasets
- *We trained the quantum diffusion model with fewer than 100 images, demonstrating the advantage of quantum layers in low-data regimes.*

## 🚀 Models

### Diffusion Model Architecture
- **U-Net backbone** with residual blocks and skip connections
- **Flexible channels** for both MNIST (grayscale) and PathMNIST (RGB)
- **Timestep embedding** using sinusoidal positional encoding
- **Cosine beta scheduling** for improved sampling
- **Exponential Moving Average (EMA)** for stable training

### Quantum Enhancement
- **Hybrid quantum-classical model** with quantum attention layers
- **Parameterized quantum circuits** implemented using PennyLane
- **RY and RZ rotations** with CNOT entanglement structure
- **Quantum feature re-weighting** mechanism

## 💿 Datasets

### MNIST
- Standard handwritten digit recognition dataset
- Trained on individual digit classes (0-9)
- Grayscale images (1-channel, 28×28)

### PathMNIST
- Medical imaging dataset from MedMNIST collection
- Colorectal cancer histology patches
- RGB images (3-channel, 28×28)
- Class-conditional training

## 📊 Results

### Training Progression Comparison

The following GIFs demonstrate the training progression of both classical and quantum diffusion models for each MNIST digit. Notice how the models learn to generate increasingly refined digit representations over 30 epochs:

#### Digit 0
<table>
  <tr>
    <td><b>Classical Model</b></td>
    <td><b>Quantum Model</b></td>
  </tr>
  <tr>
    <td><img src="result_gif/classical_mnist0_training_progress.gif" width="250"></td>
    <td><img src="result_gif/quantum_mnist0_training_progress.gif" width="250"></td>
  </tr>
</table>

#### Digit 1
<table>
  <tr>
    <td><b>Classical Model</b></td>
    <td><b>Quantum Model</b></td>
  </tr>
  <tr>
    <td><img src="result_gif/classical_mnist1_training_progress.gif" width="250"></td>
    <td><img src="result_gif/quantum_mnist1_training_progress.gif" width="250"></td>
  </tr>
</table>

#### Digit 2
<table>
  <tr>
    <td><b>Classical Model</b></td>
    <td><b>Quantum Model</b></td>
  </tr>
  <tr>
    <td><img src="result_gif/classical_mnist2_training_progress.gif" width="250"></td>
    <td><img src="result_gif/quantum_mnist2_training_progress.gif" width="250"></td>
  </tr>
</table>

#### Digit 3
<table>
  <tr>
    <td><b>Classical Model</b></td>
    <td><b>Quantum Model</b></td>
  </tr>
  <tr>
    <td><img src="result_gif/classical_mnist3_training_progress.gif" width="250"></td>
    <td><img src="result_gif/quantum_mnist3_training_progress.gif" width="250"></td>
  </tr>
</table>

#### Digit 4
<table>
  <tr>
    <td><b>Classical Model</b></td>
    <td><b>Quantum Model</b></td>
  </tr>
  <tr>
    <td><img src="result_gif/classical_mnist4_training_progress.gif" width="250"></td>
    <td><img src="result_gif/quantum_mnist4_training_progress.gif" width="250"></td>
  </tr>
</table>

#### Digit 5
<table>
  <tr>
    <td><b>Classical Model</b></td>
    <td><b>Quantum Model</b></td>
  </tr>
  <tr>
    <td><img src="result_gif/classical_mnist5_training_progress.gif" width="250"></td>
    <td><img src="result_gif/quantum_mnist5_training_progress.gif" width="250"></td>
  </tr>
</table>

#### Digit 6
<table>
  <tr>
    <td><b>Classical Model</b></td>
    <td><b>Quantum Model</b></td>
  </tr>
  <tr>
    <td><img src="result_gif/classical_mnist6_training_progress.gif" width="250"></td>
    <td><img src="result_gif/quantum_mnist6_training_progress.gif" width="250"></td>
  </tr>
</table>

#### Digit 7
<table>
  <tr>
    <td><b>Classical Model</b></td>
    <td><b>Quantum Model</b></td>
  </tr>
  <tr>
    <td><img src="result_gif/classical_mnist7_training_progress.gif" width="250"></td>
    <td><img src="result_gif/quantum_mnist7_training_progress.gif" width="250"></td>
  </tr>
</table>

#### Digit 8
<table>
  <tr>
    <td><b>Classical Model</b></td>
    <td><b>Quantum Model</b></td>
  </tr>
  <tr>
    <td><img src="result_gif/classical_mnist8_training_progress.gif" width="250"></td>
    <td><img src="result_gif/quantum_mnist8_training_progress.gif" width="250"></td>
  </tr>
</table>

#### Digit 9
<table>
  <tr>
    <td><b>Classical Model</b></td>
    <td><b>Quantum Model</b></td>
  </tr>
  <tr>
    <td><img src="result_gif/classical_mnist9_training_progress.gif" width="250"></td>
    <td><img src="result_gif/quantum_mnist9_training_progress.gif" width="250"></td>
  </tr>
</table>

### Quantitative Evaluation

The project evaluates generated images using:
- **Fréchet Inception Distance (FID)**: measures the similarity between generated and real image distributions
- **Structural Similarity Index (SSIM)**: measures the perceptual difference between images

Sample results comparing classical and quantum models:
| Model | Dataset | FID↓ | SSIM↑ |
|-------|---------|------|-------|
| Classical | MNIST | 271.05 | 0.1085 |
| Quantum | MNIST | 259.25 | 0.1263 |
| Classical | PathMNIST | 95.72 | 0.4107 |
| Quantum | PathMNIST | 84.40 | 0.0931 |

## 🔧 Implementation

### Training
```python
# Train classical diffusion model on MNIST
python quantum_difussion_mnist_v7.py  # --use_quantum=False

# Train quantum diffusion model on MNIST
python quantum_difussion_mnist_v7.py  # --use_quantum=True

# Train on PathMNIST
python quantum_difussion_pathmnist_v7.py  # --use_quantum=True/False
```

### Evaluation
```python
# Evaluate generated PathMNIST samples
python cal_fid_ssim_medmnist.py

# Evaluate generated MNIST samples
python cal_fid_ssim.py

# Debug image splitting for FID calculation
python debug_img.py
```

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/quantum-diffusion.git
cd quantum-diffusion

# Create a conda environment
conda create -n quantum-diffusion python=3.8
conda activate quantum-diffusion

# Install dependencies
pip install torch torchvision tqdm matplotlib pennylane medmnist scikit-image scipy
```

## 📝 Citation

If you use this code for your research, please cite:

```
@misc{quantum-diffusion,
  author = {Your Name},
  title = {Quantum-Enhanced Diffusion Models},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/quantum-diffusion}}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 
