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

def calculate_metrics(secret_path, recovered_path):
    """Calculate PSNR and SSIM between original secret and recovered secret images"""
    # Load images
    secret_tensor = load_image_for_metrics(secret_path)
    recovered_tensor = load_image_for_metrics(recovered_path)
    
    # Calculate PSNR (Peak Signal-to-Noise Ratio)
    psnr_value = calculate_psnr(secret_tensor, recovered_tensor).item()
    
    # Calculate SSIM (Structural Similarity Index)
    ssim_value = calculate_ssim(secret_tensor, recovered_tensor).item()
    
    return psnr_value, ssim_value


if __name__ == "__main__":
    import sys
    
    # Paths to your images
    if len(sys.argv) >= 3:
        secret_path = sys.argv[1]
        recovered_path = sys.argv[2]
    else:
        secret_path = "hide.png"
        recovered_path = "recovered_secret_new.png"
    
    # Calculate metrics
    psnr, ssim = calculate_metrics(secret_path, recovered_path)
    
    print(f"Recovery Metrics between {secret_path} and {recovered_path}:")
    print(f"PSNR: {psnr:.4f} dB")
    print(f"SSIM: {ssim:.4f}")
    
    print("\nInterpretation:")
    print(f"- PSNR: Higher values (typically >30dB) indicate better recovery quality")
    print(f"- SSIM: Values closer to 1.0 indicate higher similarity between original and recovered")