import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

# Import the model architecture
from dwt_vs_ilwt_comparison_224 import StarINNWithILWT, rgb_to_ycbcr, ycbcr_to_rgb


def load_image(path, size):
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
    parser = argparse.ArgumentParser(description="Embed a secret image into a cover image using a trained ILWT model")
    parser.add_argument("--model", required=True, help="Path to trained .pth file")
    parser.add_argument("--cover", required=True, help="Path to cover image (PNG/JPG)")
    parser.add_argument("--secret", required=True, help="Path to secret image (PNG/JPG)")
    parser.add_argument("--output", required=True, help="Output stego image path (PNG)")
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

    # Load and prepare inputs
    cover_tensor = load_image(args.cover, args.size)
    secret_tensor = load_image(args.secret, args.size)
    input_tensor = torch.cat([cover_tensor, secret_tensor], dim=0).unsqueeze(0).to(device)

    with torch.no_grad():
        stego_output, _ = model(input_tensor)
        stego_host = stego_output[:, :3, :, :][0]

    # Save stego image
    stego_img = denormalize_to_pil(stego_host)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    stego_img.save(args.output)
    print(f"Saved stego image to {args.output}")

    # No .pt tensor saved in stego-only mode


if __name__ == "__main__":
    main()


