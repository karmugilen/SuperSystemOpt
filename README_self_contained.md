# Self-Contained Embedding and Extraction Tools

This repository contains self-contained Python scripts for embedding and extracting secret images using a trained model, without requiring the training file.

## Files
- `embed_self_contained.py` - Embeds a secret image into a cover image
- `extract_self_contained.py` - Extracts a secret image from a stego image

## Prerequisites
- Python 3.7+
- PyTorch
- Pillow (PIL)
- torchvision
- numpy

## Usage

### Embedding
```bash
python embed_self_contained.py \
  --model path/to/your/model.pth \
  --cover path/to/cover/image.png \
  --secret path/to/secret/image.png \
  --output path/to/output/stego.png \
  --size 224 \
  --num_blocks 6 \
  --hidden_channels 96 \
  --transform_type ilwt53
```

### Extraction
```bash
python extract_self_contained.py \
  --model path/to/your/model.pth \
  --stego path/to/stego/image.png \
  --output path/to/output/recovered_secret.png \
  --size 224 \
  --num_blocks 6 \
  --hidden_channels 96 \
  --transform_type ilwt53
```

## Arguments
- `--model` (required): Path to the trained .pth file
- `--size`: Square size for preprocessing (default: 224)
- `--num_blocks`: Number of StarINN blocks (must match training)
- `--hidden_channels`: Hidden channels in coupling nets (match training)
- `--transform_type`: Transform backend used in training (default: "ilwt53", options: "ilwt53", "haar_conv")
- `--kY`: Y channel scaling factor (default: 0.01)
- `--kC`: Cb/Cr channel scaling factor (default: 0.04)

## Note
These scripts are completely self-contained - they include all the necessary model definitions and utility functions so you don't need the original training file (dwt_vs_ilwt_comparison_224.py).