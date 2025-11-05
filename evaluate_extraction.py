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


def calculate_metrics(original_path, recovered_path):
    """Calculate PSNR and SSIM between original and recovered images"""
    # Load images
    original_tensor = load_image_for_metrics(original_path)
    recovered_tensor = load_image_for_metrics(recovered_path)
    
    # Calculate PSNR (Peak Signal-to-Noise Ratio)
    psnr_value = calculate_psnr(recovered_tensor, original_tensor).item()
    
    # Calculate SSIM (Structural Similarity Index)
    ssim_value = calculate_ssim(recovered_tensor, original_tensor).item()
    
    return psnr_value, ssim_value


if __name__ == "__main__":
    # Paths to your images
    original_secret_path = "hide.png"
    recovered_secret_path = "recovered_secret.png"
    
    # Calculate metrics
    psnr, ssim = calculate_metrics(original_secret_path, recovered_secret_path)
    
    print(f"Quality Metrics between {original_secret_path} and {recovered_secret_path}:")
    print(f"PSNR: {psnr:.4f} dB")
    print(f"SSIM: {ssim:.4f}")
    
    print("\nInterpretation:")
    print(f"- PSNR: Higher values (typically >30dB) indicate better quality recovery")
    print(f"- SSIM: Values closer to 1.0 indicate higher similarity between original and recovered")