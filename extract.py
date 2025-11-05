import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

# Import the model architecture
from dwt_vs_ilwt_comparison_224 import StarINNWithILWT, rgb_to_ycbcr, ycbcr_to_rgb


def load_image_for_tensor(path, size):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = Image.open(path).convert('RGB')
    tensor = transform(img)
    return tensor


def denormalize_to_pil(tensor):
    tensor = (tensor / 2.0) + 0.5
    tensor = torch.clamp(tensor, 0.0, 1.0)
    img = transforms.ToPILImage()(tensor.cpu())
    return img


def main():
    parser = argparse.ArgumentParser(description="Extract a secret image from a stego image using a trained ILWT model")
    parser.add_argument("--model", required=True, help="Path to trained .pth file")
    parser.add_argument("--stego", required=True, help="Path to stego image (PNG/JPG)")
    parser.add_argument("--output", required=True, help="Output recovered secret image path (PNG)")
    parser.add_argument("--size", type=int, default=224, help="Square size for preprocessing (default: 224)")
    parser.add_argument("--num_blocks", type=int, default=6, help="Number of StarINN blocks (must match training)")
    parser.add_argument("--hidden_channels", type=int, default=96, help="Hidden channels in coupling nets (match training)")
    parser.add_argument("--transform_type", type=str, default="ilwt53", choices=["ilwt53", "haar_conv"], help="Transform backend used in training")
    parser.add_argument("--kY", type=float, default=0.01, help="Y channel scaling factor (default: 0.01)")
    parser.add_argument("--kC", type=float, default=0.04, help="Cb/Cr channel scaling factor (default: 0.04)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    # Load model
    model = StarINNWithILWT(channels=6, num_blocks=args.num_blocks, hidden_channels=args.hidden_channels, transform_type=args.transform_type)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    # Prepare stego-like input from PNG: [stego, zeros]
    stego_tensor = load_image_for_tensor(args.stego, args.size).unsqueeze(0).to(device)
    stego_like = torch.cat([stego_tensor, torch.zeros_like(stego_tensor)], dim=1)

    with torch.no_grad():
        reconstructed_input = model.inverse(stego_like)
        recovered_secret = reconstructed_input[:, 3:, :, :][0]

    recovered_img = denormalize_to_pil(recovered_secret)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    recovered_img.save(args.output)
    print(f"Saved recovered secret to {args.output}")


if __name__ == "__main__":
    main()


