import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from PIL import Image
from dwt_vs_ilwt_comparison_224 import StarINNWithILWT, calculate_psnr, calculate_ssim, calculate_bit_acc_and_bpp, ImageSteganographyDataset

def comprehensive_research_evaluation(model_path, dataset, num_samples=20, num_blocks=6, hidden_channels=96):
    """
    Comprehensive evaluation for research paper with all required metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = StarINNWithILWT(
        channels=6,
        num_blocks=num_blocks,
        hidden_channels=hidden_channels,
        transform_type="ilwt53",
    )
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    
    print(f"Comprehensive Research Evaluation with {num_samples} samples...")
    
    # Collect metrics for all samples
    all_metrics = {
        'sample_ids': [],
        'hiding_psnr': [],
        'recovery_psnr': [],
        'hiding_ssim': [],
        'recovery_ssim': [],
        'bit_acc': [],
        'hiding_mse': [],
        'recovery_mse': [],
        'extraction_success_rate_90': [],  # bit_acc >= 0.9
        'extraction_success_rate_95': [],  # bit_acc >= 0.95
        'perceptual_quality': [],
        'bpp': 24.0  # This is constant based on input dimensions
    }
    
    torch.set_grad_enabled(False)
    
    test_indices = list(range(min(num_samples, len(dataset))))
    
    for i, idx in enumerate(test_indices):
        print(f"Processing sample {i+1}/{num_samples}...")
        
        input_tensor, host_tensor, secret_tensor = dataset[idx]
        input_tensor = input_tensor.unsqueeze(0).to(device)
        host_tensor = host_tensor.unsqueeze(0).to(device)
        secret_tensor = secret_tensor.unsqueeze(0).to(device)

        # Forward pass - create stego image
        stego_output, _ = model(input_tensor)

        # Inverse pass - recover secret using stego-only path
        stego_host = stego_output[:, :3, :, :]
        stego_like = torch.cat([stego_host, torch.zeros_like(stego_host)], dim=1)
        reconstructed_input = model.inverse(stego_like)
        recovered_secret = reconstructed_input[:, 3:, :, :]

        # Calculate all metrics
        hiding_psnr = calculate_psnr(stego_host, host_tensor)
        recovery_psnr = calculate_psnr(recovered_secret, secret_tensor)
        hiding_ssim = calculate_ssim(stego_host, host_tensor)
        recovery_ssim = calculate_ssim(recovered_secret, secret_tensor)
        bit_acc, bpp = calculate_bit_acc_and_bpp(recovered_secret, secret_tensor)
        
        # Calculate MSE
        hiding_mse = torch.nn.functional.mse_loss(stego_host, host_tensor)
        recovery_mse = torch.nn.functional.mse_loss(recovered_secret, secret_tensor)
        
        # Calculate perceptual quality index (simplified)
        perceptual_quality = recovery_ssim.item() * (recovery_psnr.item() / 100.0)  # Normalized approach

        # Store metrics
        all_metrics['sample_ids'].append(idx)
        all_metrics['hiding_psnr'].append(hiding_psnr.item())
        all_metrics['recovery_psnr'].append(recovery_psnr.item())
        all_metrics['hiding_ssim'].append(hiding_ssim.item())
        all_metrics['recovery_ssim'].append(recovery_ssim.item())
        all_metrics['bit_acc'].append(bit_acc)
        all_metrics['hiding_mse'].append(hiding_mse.item())
        all_metrics['recovery_mse'].append(recovery_mse.item())
        all_metrics['extraction_success_rate_90'].append(1 if bit_acc >= 0.9 else 0)
        all_metrics['extraction_success_rate_95'].append(1 if bit_acc >= 0.95 else 0)
        all_metrics['perceptual_quality'].append(perceptual_quality)

    # Calculate averages and statistics
    avg_metrics = {
        'avg_hiding_psnr': np.mean(all_metrics['hiding_psnr']),
        'std_hiding_psnr': np.std(all_metrics['hiding_psnr']),
        'min_hiding_psnr': np.min(all_metrics['hiding_psnr']),
        'max_hiding_psnr': np.max(all_metrics['hiding_psnr']),
        
        'avg_recovery_psnr': np.mean(all_metrics['recovery_psnr']),
        'std_recovery_psnr': np.std(all_metrics['recovery_psnr']),
        'min_recovery_psnr': np.min(all_metrics['recovery_psnr']),
        'max_recovery_psnr': np.max(all_metrics['recovery_psnr']),
        
        'avg_hiding_ssim': np.mean(all_metrics['hiding_ssim']),
        'std_hiding_ssim': np.std(all_metrics['hiding_ssim']),
        'min_hiding_ssim': np.min(all_metrics['hiding_ssim']),
        'max_hiding_ssim': np.max(all_metrics['hiding_ssim']),
        
        'avg_recovery_ssim': np.mean(all_metrics['recovery_ssim']),
        'std_recovery_ssim': np.std(all_metrics['recovery_ssim']),
        'min_recovery_ssim': np.min(all_metrics['recovery_ssim']),
        'max_recovery_ssim': np.max(all_metrics['recovery_ssim']),
        
        'avg_bit_acc': np.mean(all_metrics['bit_acc']),
        'std_bit_acc': np.std(all_metrics['bit_acc']),
        'min_bit_acc': np.min(all_metrics['bit_acc']),
        'max_bit_acc': np.max(all_metrics['bit_acc']),
        
        'avg_hiding_mse': np.mean(all_metrics['hiding_mse']),
        'std_hiding_mse': np.std(all_metrics['hiding_mse']),
        
        'avg_recovery_mse': np.mean(all_metrics['recovery_mse']),
        'std_recovery_mse': np.std(all_metrics['recovery_mse']),
        
        'avg_perceptual_quality': np.mean(all_metrics['perceptual_quality']),
        'std_perceptual_quality': np.std(all_metrics['perceptual_quality']),
        
        'extraction_success_rate_90': np.mean(all_metrics['extraction_success_rate_90']),
        'extraction_success_rate_95': np.mean(all_metrics['extraction_success_rate_95']),
        
        'bpp': all_metrics['bpp'],
        'total_samples': len(all_metrics['sample_ids'])
    }
    
    # Print research results summary
    print("\n" + "="*80)
    print("COMPREHENSIVE RESEARCH EVALUATION RESULTS")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Samples Evaluated: {avg_metrics['total_samples']}")
    print(f"Embedding Capacity: {avg_metrics['bpp']:.2f} bpp")
    print("-"*80)
    print(f"Hiding Quality Metrics:")
    print(f"  PSNR: {avg_metrics['avg_hiding_psnr']:.2f}±{avg_metrics['std_hiding_psnr']:.2f} dB (Range: {avg_metrics['min_hiding_psnr']:.2f}-{avg_metrics['max_hiding_psnr']:.2f})")
    print(f"  SSIM: {avg_metrics['avg_hiding_ssim']:.4f}±{avg_metrics['std_hiding_ssim']:.4f} (Range: {avg_metrics['min_hiding_ssim']:.4f}-{avg_metrics['max_hiding_ssim']:.4f})")
    print()
    print(f"Recovery Quality Metrics:")
    print(f"  PSNR: {avg_metrics['avg_recovery_psnr']:.2f}±{avg_metrics['std_recovery_psnr']:.2f} dB (Range: {avg_metrics['min_recovery_psnr']:.2f}-{avg_metrics['max_recovery_psnr']:.2f})")
    print(f"  SSIM: {avg_metrics['avg_recovery_ssim']:.4f}±{avg_metrics['std_recovery_ssim']:.4f} (Range: {avg_metrics['min_recovery_ssim']:.4f}-{avg_metrics['max_recovery_ssim']:.4f})")
    print()
    print(f"Extraction Accuracy:")
    print(f"  Bit Accuracy: {avg_metrics['avg_bit_acc']:.4f}±{avg_metrics['std_bit_acc']:.4f} (Range: {avg_metrics['min_bit_acc']:.4f}-{avg_metrics['max_bit_acc']:.4f})")
    print(f"  Success Rate (≥90%): {avg_metrics['extraction_success_rate_90']:.4f}")
    print(f"  Success Rate (≥95%): {avg_metrics['extraction_success_rate_95']:.4f}")
    print()
    print(f"Perceptual Quality: {avg_metrics['avg_perceptual_quality']:.4f}±{avg_metrics['std_perceptual_quality']:.4f}")
    print("="*80)
    
    # Create comprehensive research plots
    create_research_plots(all_metrics, avg_metrics)
    
    # Save detailed metrics
    os.makedirs("research_metrics", exist_ok=True)
    with open("research_metrics/comprehensive_evaluation.json", "w") as f:
        json.dump({
            "all_sample_metrics": all_metrics,
            "average_metrics": avg_metrics
        }, f, indent=4)
    
    print("\nDetailed metrics saved to research_metrics/comprehensive_evaluation.json")
    print("Research plots saved to research_plots/ directory")
    
    return avg_metrics, all_metrics

from dwt_vs_ilwt_comparison_224 import split_dataset

def create_research_plots(all_metrics, avg_metrics):
    """Create comprehensive plots for research paper"""
    os.makedirs("research_plots", exist_ok=True)
    
    fig = plt.figure(figsize=(20, 24))
    
    # Plot 1: PSNR Distribution
    plt.subplot(4, 3, 1)
    plt.hist(all_metrics['hiding_psnr'], bins=20, alpha=0.7, label='Hiding PSNR', color='g')
    plt.hist(all_metrics['recovery_psnr'], bins=20, alpha=0.7, label='Recovery PSNR', color='r')
    plt.title('PSNR Distribution', fontweight='bold')
    plt.xlabel('PSNR (dB)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: SSIM Distribution
    plt.subplot(4, 3, 2)
    plt.hist(all_metrics['hiding_ssim'], bins=20, alpha=0.7, label='Hiding SSIM', color='g')
    plt.hist(all_metrics['recovery_ssim'], bins=20, alpha=0.7, label='Recovery SSIM', color='r')
    plt.title('SSIM Distribution', fontweight='bold')
    plt.xlabel('SSIM')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Bit Accuracy Distribution
    plt.subplot(4, 3, 3)
    plt.hist(all_metrics['bit_acc'], bins=20, alpha=0.7, color='c')
    plt.title('Bit Accuracy Distribution', fontweight='bold')
    plt.xlabel('Bit Accuracy')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: PSNR Correlation
    plt.subplot(4, 3, 4)
    plt.scatter(all_metrics['hiding_psnr'], all_metrics['recovery_psnr'], alpha=0.6)
    plt.title('Hiding PSNR vs Recovery PSNR', fontweight='bold')
    plt.xlabel('Hiding PSNR (dB)')
    plt.ylabel('Recovery PSNR (dB)')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: SSIM Correlation
    plt.subplot(4, 3, 5)
    plt.scatter(all_metrics['hiding_ssim'], all_metrics['recovery_ssim'], alpha=0.6, color='orange')
    plt.title('Hiding SSIM vs Recovery SSIM', fontweight='bold')
    plt.xlabel('Hiding SSIM')
    plt.ylabel('Recovery SSIM')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Bit Accuracy vs Recovery PSNR
    plt.subplot(4, 3, 6)
    plt.scatter(all_metrics['bit_acc'], all_metrics['recovery_psnr'], alpha=0.6, color='purple')
    plt.title('Bit Accuracy vs Recovery PSNR', fontweight='bold')
    plt.xlabel('Bit Accuracy')
    plt.ylabel('Recovery PSNR (dB)')
    plt.grid(True, alpha=0.3)
    
    # Plot 7: Sample-wise metrics
    plt.subplot(4, 3, 7)
    x = range(len(all_metrics['hiding_psnr']))
    plt.plot(x, all_metrics['hiding_psnr'], label='Hiding PSNR', marker='o', linestyle='-', markersize=4)
    plt.plot(x, all_metrics['recovery_psnr'], label='Recovery PSNR', marker='s', linestyle='-', markersize=4)
    plt.title('Sample-wise PSNR Metrics', fontweight='bold')
    plt.xlabel('Sample Index')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 8: Sample-wise SSIM
    plt.subplot(4, 3, 8)
    x = range(len(all_metrics['hiding_ssim']))
    plt.plot(x, all_metrics['hiding_ssim'], label='Hiding SSIM', marker='o', linestyle='-', markersize=4)
    plt.plot(x, all_metrics['recovery_ssim'], label='Recovery SSIM', marker='s', linestyle='-', markersize=4)
    plt.title('Sample-wise SSIM Metrics', fontweight='bold')
    plt.xlabel('Sample Index')
    plt.ylabel('SSIM')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 9: Sample-wise Bit Accuracy
    plt.subplot(4, 3, 9)
    x = range(len(all_metrics['bit_acc']))
    plt.plot(x, all_metrics['bit_acc'], label='Bit Accuracy', marker='o', linestyle='-', markersize=4, color='c')
    plt.axhline(y=0.9, color='r', linestyle='--', label='90% Threshold', alpha=0.7)
    plt.axhline(y=0.95, color='m', linestyle='--', label='95% Threshold', alpha=0.7)
    plt.title('Sample-wise Bit Accuracy', fontweight='bold')
    plt.xlabel('Sample Index')
    plt.ylabel('Bit Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 10: MSE Distribution
    plt.subplot(4, 3, 10)
    plt.hist(all_metrics['hiding_mse'], bins=20, alpha=0.7, label='Hiding MSE', color='g')
    plt.hist(all_metrics['recovery_mse'], bins=20, alpha=0.7, label='Recovery MSE', color='r')
    plt.title('MSE Distribution', fontweight='bold')
    plt.xlabel('MSE')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 11: Success Rate Analysis
    plt.subplot(4, 3, 11)
    success_rates = [avg_metrics['extraction_success_rate_90'], avg_metrics['extraction_success_rate_95']]
    labels = ['≥90% Acc', '≥95% Acc']
    colors = ['lightblue', 'lightgreen']
    bars = plt.bar(labels, success_rates, color=colors)
    plt.title('Extraction Success Rates', fontweight='bold')
    plt.ylabel('Proportion')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{rate:.3f}', ha='center', va='bottom')
    plt.grid(True, alpha=0.3)
    
    # Plot 12: Perceptual Quality
    plt.subplot(4, 3, 12)
    plt.hist(all_metrics['perceptual_quality'], bins=20, alpha=0.7, color='orange')
    plt.title('Perceptual Quality Distribution', fontweight='bold')
    plt.xlabel('Perceptual Quality Index')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("research_plots/comprehensive_research_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Comprehensive research plots saved!")

if __name__ == "__main__":
    # Run comprehensive evaluation
    image_dir = "my_images"
    img_size = 224
    dataset = ImageSteganographyDataset(image_dir, img_size=img_size)
    
    # Run evaluation
    avg_metrics, all_metrics = comprehensive_research_evaluation(
        model_path="ilwt_steganography_model.pth",
        dataset=dataset,
        num_samples=20
    )