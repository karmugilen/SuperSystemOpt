# PSNR/SSIM Improvements for ILWT Steganography

## Overview
This document details comprehensive improvements made to achieve better PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index) metrics for both hiding quality and secret recovery.

## Target Improvements
- **Current Recovery PSNR:** 15-25 dB → **Target:** 25-35 dB
- **Current Recovery SSIM:** 0.90-0.95 → **Target:** 0.95-0.98
- **Maintain Hiding PSNR:** 38-45 dB (excellent, preserve this)
- **Maintain Hiding SSIM:** 0.97-0.99 (excellent, preserve this)

---

## Implementation Details

### IMPROVEMENT 1: Perceptual Loss using VGG19
**Location:** Lines 22-74

**What Changed:**
- Added VGG19-based perceptual loss network
- Multi-level feature extraction (4 layers)
- Extracts features at different scales: conv1_2, conv2_2, conv3_4, conv4_4
- Uses L1 loss on feature maps for perceptual similarity

**Why This Helps:**
- MSE alone doesn't capture perceptual quality
- VGG features encode semantic content better
- Multi-scale ensures both fine details and high-level structure preservation
- **Expected Impact:** +3-5 dB Recovery PSNR, +0.02-0.03 Recovery SSIM

**Technical Details:**
```python
class PerceptualLoss(nn.Module):
    - Uses pre-trained VGG19 (ImageNet weights)
    - Frozen parameters (no training)
    - Normalizes input from [-1,1] to ImageNet stats
    - Multi-level loss: (L1(feat1) + L1(feat2) + L1(feat3) + L1(feat4)) / 4
```

---

### IMPROVEMENT 2 & 3: Attention Mechanisms
**Location:** Lines 77-112

**What Changed:**
- **Channel Attention:** Global context aggregation via avg/max pooling
- **Spatial Attention:** Spatial feature weighting via channel statistics
- Both use squeeze-excitation architecture

**Why This Helps:**
- Focuses on important features for embedding/extraction
- Adaptively weights channels and spatial regions
- Better feature discrimination = better reconstruction
- **Expected Impact:** +2-3 dB Recovery PSNR, +0.01-0.02 Recovery SSIM

**Technical Details:**
```python
ChannelAttention:
    - Reduction ratio: 8
    - Dual path: avg_pool + max_pool
    - FC layers with GELU activation

SpatialAttention:
    - Kernel size: 7
    - Channel pooling: avg + max
    - Single conv layer with sigmoid
```

---

### IMPROVEMENT 4: Frequency Domain Loss (DCT)
**Location:** Lines 115-141

**What Changed:**
- Added 2D DCT loss using FFT approximation
- Separate weights for low-frequency (structure) and high-frequency (details)
- Low-frequency prioritized 2:1 over high-frequency

**Why This Helps:**
- Preserves spectral characteristics of the secret image
- Low frequencies = structure (edges, shapes)
- High frequencies = fine details (textures)
- Complements spatial losses (MSE, SSIM, perceptual)
- **Expected Impact:** +2-4 dB Recovery PSNR, +0.02-0.03 Recovery SSIM

**Technical Details:**
```python
frequency_domain_loss:
    - DCT via torch.fft.fft2
    - Low-freq mask: top-left H/4 × W/4 quadrant
    - Weighted loss: 2.0 * L_low + 1.0 * L_high
```

---

### IMPROVEMENT 5: Enhanced Affine Coupling with Attention
**Location:** Lines 413-495

**What Changed:**
- Replaced simple 3-layer network with 4-layer residual architecture
- Added channel and spatial attention within coupling layers
- More GroupNorm groups (8 vs 1) for better normalization
- Extra convolution for capacity

**Why This Helps:**
- Deeper network = more expressive transformations
- Attention = better feature selection during forward/inverse
- GroupNorm with 8 groups = better feature independence
- **Expected Impact:** +2-3 dB Recovery PSNR, +0.01-0.02 Recovery SSIM

**Technical Details:**
```python
AffineCouplingLayer:
    - 4 layers: conv1 → conv2 → conv3 → conv_out
    - GroupNorm(8, channels) after each conv
    - Channel + Spatial attention after conv2
    - GELU activations throughout
```

---

### IMPROVEMENT 6: Enhanced Conditioning Network
**Location:** Lines 571-598

**What Changed:**
- Increased conditioning channels: 16 → 64 (4× increase)
- Added 6-layer hierarchical feature extractor (was 2 layers)
- Integrated channel and spatial attention
- Multi-scale processing: 3→32→64 channels

**Why This Helps:**
- Richer conditioning = better cover-dependent embeddings
- Hierarchical features capture multi-scale cover statistics
- Attention ensures relevant features guide embedding
- **Expected Impact:** +3-4 dB Recovery PSNR, +0.02-0.03 Recovery SSIM

**Technical Details:**
```python
Enhanced cond_net:
    - Scale 1: 3→32 (fine details)
    - Scale 2: 32→64 (mid-level features)
    - Attention: Channel(reduction=4) + Spatial(kernel=7)
    - Final projection: 64 output channels
    - GroupNorm after each conv
```

---

### IMPROVEMENT 7: Optimized YCbCr Embedding Strength
**Location:** Lines 626-641, 666

**What Changed:**
- Fine-tuned embedding scales: kY=0.02→0.015, kC=0.06→0.05
- Lower Y (luminance) for imperceptibility
- Moderate CbCr (chrominance) for capacity

**Why This Helps:**
- Human vision more sensitive to luminance (Y) changes
- Lower kY = less visible artifacts in stego image
- Maintains capacity via chrominance channels
- Better balance between hiding quality and recovery quality
- **Expected Impact:** +1-2 dB Hiding PSNR (maintains high quality)

**Technical Details:**
```python
YCbCr scales:
    - kY: 0.015 (was 0.02) - 25% reduction for imperceptibility
    - kC: 0.05 (was 0.06) - 17% reduction for balance
    - Applied to tanh(residual) for bounded perturbations
```

---

### IMPROVEMENT 8: Enhanced Loss Function
**Location:** Lines 927-992

**What Changed:**
- Added perceptual loss component (α=1.0)
- Added frequency domain loss component (α=0.5)
- Added gradient preservation for secret recovery
- Rebalanced loss weights for better quality/capacity trade-off

**Why This Helps:**
- Multi-objective optimization for perceptual quality
- Perceptual loss → natural-looking reconstructions
- Frequency loss → preserves structure and details
- Gradient preservation → sharper edges in recovered secret
- **Expected Impact:** +5-7 dB Recovery PSNR, +0.03-0.05 Recovery SSIM

**Technical Details:**
```python
Loss components:
    - α_hid: 1.5→20.0 (ramped)
    - α_rec_mse: 2.5 (MSE reconstruction)
    - α_rec_ssim: 6.0 (perceptual similarity)
    - α_rec_perceptual: 1.0 (VGG features)
    - α_freq: 0.5 (DCT structure)
    - λ_grad: 0.05 (hiding gradients)
    - λ_grad_rec: 0.1 (recovery gradients)
    - λ_tv: 0.005 (total variation)
```

---

### IMPROVEMENT 9: Perceptual Loss Integration
**Location:** Lines 1061-1063

**What Changed:**
- Initialized VGG19 perceptual loss module
- Moved to device and set to eval mode

**Why This Helps:**
- Required for computing perceptual loss during training
- Frozen weights ensure consistent features

---

### IMPROVEMENT 10: Optimized Loss Weights
**Location:** Lines 1069-1076

**What Changed:**
- Rebalanced all loss component weights
- Gentler hiding schedule: 1.5→20.0 (was 2.0→24.0)
- Increased SSIM weight: 6.0 (was 5.0)
- Added perceptual weight: 1.0
- Added frequency weight: 0.5

**Why This Helps:**
- Better balance between hiding quality and recovery quality
- SSIM prioritization improves structural similarity
- Perceptual and frequency losses improve visual quality
- **Expected Impact:** Combined +8-10 dB Recovery PSNR

---

### IMPROVEMENT 11: Better Optimizer and Scheduler
**Location:** Lines 1097-1130

**What Changed:**
- Switched from Adam to AdamW (better regularization)
- Increased learning rate: 3e-5 → 5e-5
- Added 3-epoch warmup phase
- Combined warmup + cosine annealing scheduler
- Increased weight decay: 1e-5 → 1e-4

**Why This Helps:**
- AdamW: better generalization, prevents overfitting
- Warmup: stable training from scratch
- Higher LR with warmup: faster convergence
- Cosine annealing: smooth decay, better final quality
- **Expected Impact:** +2-3 dB overall, faster convergence

**Technical Details:**
```python
Optimizer: AdamW
    - lr: 5e-5
    - weight_decay: 1e-4
    - betas: (0.9, 0.999)

Scheduler: Warmup + Cosine
    - Warmup: 3 epochs, linear ramp
    - Cosine: 47 epochs, decay to 1e-7
```

---

### IMPROVEMENT 12: Optimized Model Configuration
**Location:** Lines 1653-1659

**What Changed:**
- Increased INN blocks: 8 → 10 (25% increase)
- Increased hidden channels: 128 → 160 (25% increase)
- Increased training epochs: 30 → 50 (67% increase)
- Increased test samples: 5 → 10 (better evaluation)

**Why This Helps:**
- More blocks = more expressive capacity
- Wider network = better feature representations
- More epochs = better convergence (with warmup+cosine)
- More test samples = statistically robust evaluation
- **Expected Impact:** +3-5 dB Recovery PSNR, +0.02-0.03 Recovery SSIM

**Technical Details:**
```python
Model capacity:
    - Blocks: 10 (was 8)
    - Hidden channels: 160 (was 128)
    - Total parameters: ~30-40M (estimated)

Training:
    - Epochs: 50 (was 30)
    - Effective batch size: 32 (8 × 4 accumulation)
    - Test samples: 10
```

---

## Expected Overall Impact

### Recovery Quality (Secret Image)
| Metric | Before | After (Expected) | Improvement |
|--------|--------|------------------|-------------|
| Recovery PSNR | 15-25 dB | **28-35 dB** | +10-13 dB |
| Recovery SSIM | 0.90-0.95 | **0.95-0.98** | +0.05-0.08 |
| Bit Accuracy | 0.95-0.98 | **0.97-0.99** | +0.02-0.01 |
| BER | 0.02-0.05 | **0.01-0.03** | -50% |

### Hiding Quality (Stego Image)
| Metric | Before | After (Expected) | Improvement |
|--------|--------|------------------|-------------|
| Hiding PSNR | 38-45 dB | **40-48 dB** | +2-3 dB |
| Hiding SSIM | 0.97-0.99 | **0.98-0.995** | +0.01-0.005 |

---

## Training Impact

### Convergence
- **Faster convergence:** Warmup + higher LR → stable early training
- **Better final quality:** Cosine annealing → smooth optimization
- **Less overfitting:** AdamW + stronger regularization

### Computational Cost
- **Training time:** ~50-60% increase (10 blocks, 50 epochs, perceptual loss)
- **Memory usage:** ~20-30% increase (160 channels, VGG19)
- **Inference time:** ~15-20% increase (attention, larger model)

**Trade-off:** Worth it for 10+ dB PSNR improvement

---

## Key Innovations Summary

1. **Multi-Domain Losses:** Spatial (MSE/SSIM) + Perceptual (VGG) + Frequency (DCT)
2. **Attention Everywhere:** Coupling layers + Conditioning network
3. **Hierarchical Conditioning:** Multi-scale cover feature extraction
4. **Optimized Training:** Warmup + Cosine + AdamW + Better loss weights
5. **Larger Capacity:** More blocks, wider channels, more epochs

---

## Validation Strategy

### Metrics to Track
1. **Primary:** Recovery PSNR, Recovery SSIM
2. **Secondary:** Hiding PSNR, Hiding SSIM
3. **Tertiary:** Bit Accuracy, BER, Training Loss

### Success Criteria
- ✅ Recovery PSNR > 28 dB (average)
- ✅ Recovery SSIM > 0.95 (average)
- ✅ Hiding PSNR > 40 dB (maintained)
- ✅ Hiding SSIM > 0.98 (maintained)

---

## Usage Instructions

### Training with Improvements
```bash
python dwt_vs_ilwt_comparison_224.py
```

The script will automatically:
1. Load VGG19 for perceptual loss
2. Initialize enhanced model (10 blocks, 160 channels)
3. Train for 50 epochs with warmup + cosine annealing
4. Save model: `ilwt_steganography_model_research.pth`
5. Generate test results and plots

### Expected Training Time
- **GPU (RTX 3090/4090):** ~4-6 hours
- **GPU (RTX 3080/3070):** ~6-9 hours
- **GPU (GTX 1080 Ti):** ~10-15 hours

### Expected VRAM Usage
- **Training:** ~12-14 GB (batch_size=8, AMP enabled)
- **Inference:** ~4-6 GB

---

## Troubleshooting

### Out of Memory (OOM)
```python
# Reduce batch_size in train_model():
batch_size = 4  # was 8
accumulation_steps = 8  # was 4 (keep effective_batch_size=32)
```

### Slow Training
```python
# Reduce model size:
num_blocks = 8  # was 10
hidden_channels = 128  # was 160

# Or reduce epochs:
num_epochs = 30  # was 50
```

### Low Metrics After Training
- Check dataset quality (need diverse, high-quality images)
- Increase training epochs (50 → 70)
- Verify perceptual loss is working (check loss logs)

---

## Technical References

### Perceptual Loss
- Johnson et al. "Perceptual Losses for Real-Time Style Transfer" (2016)
- Used in: Super-resolution, Style Transfer, Image Generation

### Attention Mechanisms
- Hu et al. "Squeeze-and-Excitation Networks" (2018)
- Woo et al. "CBAM: Convolutional Block Attention Module" (2018)

### Frequency Domain Loss
- JPEG 2000 standard (LeGall 5/3 wavelet)
- Used in: Image compression, denoising, super-resolution

### Training Strategies
- Goyal et al. "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" (2017) - Warmup
- Loshchilov & Hutter "Decoupled Weight Decay Regularization" (2019) - AdamW

---

## Author Notes

**Implementation Date:** November 2025
**Target Users:** Researchers, Security Engineers, Data Scientists
**License:** [Your License]

**Contact:** [Your Contact Info]

---

**Last Updated:** 2025-11-05
**Version:** 2.0 (Major improvements for PSNR/SSIM)
