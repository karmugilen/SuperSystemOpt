# ILWT-Based Deep Learning Image Steganography

## Overview

This project implements an **invertible neural network (INN)** for image steganography using **Integer Lifting Wavelet Transform (ILWT)**. The system can hide a complete secret image inside a cover image, producing a stego image that looks visually identical to the cover. The hidden image can later be extracted from the stego image alone, without needing the original cover or secret images.

## Key Features

- **Deep Learning Steganography**: Uses invertible neural networks for high-capacity hiding
- **ILWT (LeGall 5/3)**: Differentiable wavelet transform for frequency-domain embedding
- **High Quality**: Achieves ~38-45 dB PSNR for stego images (imperceptible to humans)
- **Full Image Hiding**: Embeds entire 224×224 RGB images (24 bits per pixel capacity)
- **Stego-Only Extraction**: Recovers hidden image using only the stego image
- **YCbCr Color Space**: Exploits human vision properties for better hiding
- **Research-Grade Metrics**: Comprehensive evaluation with PSNR, SSIM, BER, ACC, HR, BIR

## Architecture

### Model Components

1. **ILWT53_2D**: Differentiable 2D Integer Lifting Wavelet Transform
   - Forward: `(B,C,H,W) → (B,4C,H/2,W/2)` (4 subbands: LL, LH, HL, HH)
   - Inverse: Reconstructs original from wavelet coefficients
   - Perfect reconstruction with ~1e-7 error

2. **StarINNBlock**: Normalizing flow architecture
   - ActNorm: Activation normalization
   - Invertible1x1Conv: Channel mixing
   - AffineCouplingLayer: Conditional transformations

3. **StarINNWithILWT**: Complete steganography model
   - Frequency-domain embedding via ILWT
   - Cover-conditional processing
   - YCbCr color space residual composition

### How It Works

```
HIDING (Forward):
Cover Image (3ch) + Secret Image (3ch) → [6 channels]
    ↓
ILWT Transform → Frequency domain (24ch, H/2×W/2)
    ↓
StarINN Blocks (conditioned on cover features)
    ↓
Inverse ILWT → Spatial domain
    ↓
YCbCr Residual Addition (Y: 0.01, CbCr: 0.04 scale)
    ↓
Stego Image (3ch) - looks identical to cover

EXTRACTION (Inverse):
Stego Image (3ch) + Zeros (3ch) → [6 channels]
    ↓
ILWT Transform → Frequency domain
    ↓
Reverse StarINN Blocks
    ↓
Inverse ILWT → Spatial domain
    ↓
YCbCr Residual Subtraction
    ↓
Recovered Secret Image (3ch)
```

## Project Structure

```
VM/
├── dwt_vs_ilwt_comparison_224.py   # Main training/testing script
├── embed_self_contained.py         # Standalone embedding script
├── extract_self_contained.py       # Standalone extraction script
├── embed.py                         # Basic embedding (requires model)
├── extract.py                       # Basic extraction (requires model)
├── evaluate_metrics.py              # Stego image quality evaluation
├── evaluate_secret_metrics.py       # Secret recovery quality evaluation
├── research_evaluation.py           # Comprehensive research metrics
├── my_images/                       # Training dataset directory
├── ilwt_test_results/              # Test output visualizations
├── research_metrics/               # Training metrics (JSON + logs)
│   ├── training_metrics.json
│   ├── test_metrics.json
│   └── training_log.txt
├── research_plots/                 # Training visualization plots
│   ├── training_metrics.png
│   └── comprehensive_metrics.png
├── ilwt_steganography_model.pth    # Trained model weights
└── README.md                        # This file
```

## Installation

### Requirements

```bash
pip install torch torchvision numpy matplotlib pillow tqdm
```

**Dependencies:**
- Python 3.7+
- PyTorch 1.10+
- CUDA (optional, for GPU acceleration)

### Dataset Setup

Place your training images in the `my_images/` directory:
```bash
mkdir -p my_images
# Add .png, .jpg, or .jpeg images
```

## Usage

### 1. Training the Model

```bash
python dwt_vs_ilwt_comparison_224.py
```

**Configuration** (edit in `main()` function):
- `img_size = 224`: Image resolution
- `num_epochs = 3`: Training epochs (increase for production)
- `num_blocks = 6`: Number of INN blocks
- `hidden_channels = 96`: Hidden layer width
- `batch_size = 1`: Training batch size

**Outputs:**
- `ilwt_steganography_model_research.pth`: Trained weights
- `research_metrics/training_metrics.json`: Training history
- `research_metrics/training_log.txt`: Epoch-by-epoch logs
- `research_plots/*.png`: Training curves
- `ilwt_test_results/`: Test sample visualizations

### 2. Embedding a Secret Image

**Standalone (no dependencies):**
```bash
python embed_self_contained.py
```

**With trained model:**
```bash
python embed.py
```

**Input:**
- `cover.png`: Cover image (will be visible)
- `hide.png`: Secret image (will be hidden)

**Output:**
- `embedded_output.png`: Stego image (save/transmit this)

### 3. Extracting the Secret Image

**Standalone (no dependencies):**
```bash
python extract_self_contained.py
```

**With trained model:**
```bash
python extract.py
```

**Input:**
- `embedded_output.png` or `stego_output.png`: Stego image

**Output:**
- `recovered_secret.png`: Extracted secret image

### 4. Evaluation

**Stego Image Quality (vs cover):**
```bash
python evaluate_metrics.py
```

**Secret Recovery Quality (vs original secret):**
```bash
python evaluate_secret_metrics.py
```

**Comprehensive Research Metrics:**
```bash
python research_evaluation.py
```

## Metrics Explained

### Hiding Quality (Stego vs Cover)
- **PSNR**: Peak Signal-to-Noise Ratio (>35 dB = imperceptible, >40 dB = excellent)
- **SSIM**: Structural Similarity (0-1, higher better, >0.95 = good)
- **MSE**: Mean Squared Error (lower better)
- **HR**: Hiding Ratio (higher = better similarity)
- **BIR**: Bitrate Increase Ratio (lower = less distortion)

### Recovery Quality (Recovered vs Original Secret)
- **PSNR**: 15-25 dB typical for deep steganography
- **SSIM**: >0.90 indicates good recovery
- **ACC**: Bit-level accuracy (>0.95 = excellent)
- **BER**: Bit Error Rate (lower better, <0.05 = good)

### Capacity
- **BPP**: Bits Per Pixel = 24 (full RGB capacity)

## Training Details

### Loss Function

```python
Total Loss = α_hid × hiding_MSE            # Cover invisibility
            + α_rec_mse × recovery_MSE      # Secret reconstruction
            + α_rec_ssim × recovery_SSIM    # Perceptual quality
            + λ_grad × gradient_loss         # Edge preservation
            + λ_tv × total_variation         # Smoothness
            + multiscale_ILWT_loss          # Wavelet structure
```

### Curriculum Learning
- **α_hid**: Ramps from 4.0 → 48.0 (prioritizes invisibility over time)
- **Perturbations**: Gradually enabled (quantization, noise) to simulate JPEG/file I/O

### Multi-Scale Wavelet Loss
- **Level 1**: Weighted loss on LL/LH/HL/HH subbands (LL highest priority)
- **Level 2**: Additional LL-of-LL loss for deep structure preservation

## Results

### Typical Performance (after training)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Hiding PSNR | 38-45 dB | Stego visually identical to cover |
| Hiding SSIM | 0.97-0.99 | Excellent structural similarity |
| Recovery PSNR | 15-25 dB | Good secret reconstruction |
| Recovery SSIM | 0.90-0.95 | High perceptual quality |
| Bit Accuracy | 0.95-0.98 | Most bits correctly recovered |
| BER | 0.02-0.05 | Low bit error rate |

### Example Output

```
Sample 1: Hiding PSNR=41.23 dB, Recovery PSNR=22.45 dB
          Hiding SSIM=0.9823, Recovery SSIM=0.9256
          ACC=0.9612, BPP=24.00
```

## Advanced Usage

### Custom Image Sizes

Edit `_maybe_pad()` in `ILWT53_2D` to handle arbitrary dimensions (current: 224×224).

### Different Wavelet Transforms

Change `transform_type` in `StarINNWithILWT`:
- `"ilwt53"`: LeGall 5/3 (default, better quality)
- `"haar_conv"`: Learnable Haar wavelets (faster training)

### Adjusting Embedding Strength

Edit YCbCr scales in `StarINNWithILWT.forward()` (line 455):
```python
kY, kC = 0.01, 0.04  # Decrease for more invisible (lower capacity)
                      # Increase for higher capacity (more visible)
```

## Limitations

1. **Fixed Resolution**: Trained for 224×224 images (can be retrained)
2. **Lossy Compression**: JPEG compression degrades recovery (train with augmentation)
3. **Model Required**: Both sender/receiver need the trained model
4. **Compute**: GPU recommended for training (CPU for inference)

## Research Applications

- **Secure Communication**: Covert messaging
- **Copyright Protection**: Invisible watermarking
- **Data Augmentation**: Dual-purpose images
- **Steganographic Analysis**: Adversarial testing

## Citation

If you use this code in research, please cite:

```
ILWT-Based Deep Learning Image Steganography
Integer Lifting Wavelet Transform with Invertible Neural Networks
https://github.com/your-repo-here
```

## License

[Add your license here]

## References

- LeGall 5/3 Wavelet Transform (JPEG 2000)
- Normalizing Flows (Dinh et al., 2016)
- HiDDeN: Hiding Data with Deep Networks (Zhu et al., 2018)

## Troubleshooting

### "CUDA out of memory"
- Reduce `batch_size` (default: 1)
- Use smaller `hidden_channels` (default: 96)

### Low Recovery Quality
- Train for more epochs (increase `num_epochs`)
- Ensure diverse training images (>100 recommended)
- Check if stego image was compressed (use PNG, not JPEG)

### Padding Errors
- Ensure images are divisible by 2 (224×224 recommended)
- Check `_maybe_pad()` reflection padding logic

## Contact

[Add your contact/GitHub information]

---

**Last Updated:** November 2025
**Version:** 1.0
# SuperSystemOpt
