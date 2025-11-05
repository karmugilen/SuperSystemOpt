import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from dwt_vs_ilwt_comparison_224 import calculate_psnr, calculate_ssim


def load_image_for_metrics(path):
    """Load image and convert to tensor format for PSNR/SSIM calculation"""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img = Image.open(path).convert('RGB')
    tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return tensor


def calculate_metrics(cover_path, stego_path):
    """Calculate PSNR and SSIM between cover and stego images"""
    # Load images
    cover_tensor = load_image_for_metrics(cover_path)
    stego_tensor = load_image_for_metrics(stego_path)
    
    # Calculate PSNR (Peak Signal-to-Noise Ratio)
    psnr_value = calculate_psnr(cover_tensor, stego_tensor).item()
    
    # Calculate SSIM (Structural Similarity Index)
    ssim_value = calculate_ssim(cover_tensor, stego_tensor).item()
    
    return psnr_value, ssim_value


if __name__ == "__main__":
    import sys
    
    # Paths to your images
    if len(sys.argv) >= 3:
        cover_path = sys.argv[1]
        stego_path = sys.argv[2]
    else:
        cover_path = "cover.png"
        stego_path = "embedded_output.png"  # Updated to use the new stego image
    
    # Calculate metrics
    psnr, ssim = calculate_metrics(cover_path, stego_path)
    
    print(f"Quality Metrics between {cover_path} and {stego_path}:")
    print(f"PSNR: {psnr:.4f} dB")
    print(f"SSIM: {ssim:.4f}")
    
    print("\nInterpretation:")
    print(f"- PSNR: Higher values (typically >30dB) indicate better quality")
    print(f"- SSIM: Values closer to 1.0 indicate higher similarity")