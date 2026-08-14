"""
Export and Dynamic Quantization Utility for CPU Inference.
Converts PyTorch models to quantized dynamic INT8 format for ultra-fast local laptop CPU inference.
"""

import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.models.char_transformer import WordleCharTransformer, CharTransformerConfig


def export_and_quantize_char_transformer(
    checkpoint_path: str = "models/char_transformer/best_char_transformer.pt",
    output_path: str = "models/char_transformer/quantized_char_transformer.pt"
):
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} not found. Saving lightweight dummy export...")
        config = CharTransformerConfig()
        model = WordleCharTransformer(config)
    else:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        config = ckpt.get("config", CharTransformerConfig())
        model = WordleCharTransformer(config)
        model.load_state_dict(ckpt["model_state_dict"])

    model.eval()

    # Dynamic INT8 Quantization for Linear Layers
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(quantized_model.state_dict(), output_path)
    print(f"Quantized INT8 model successfully saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export & Quantize Models for CPU Inference")
    parser.add_argument("--ckpt", type=str, default="models/char_transformer/best_char_transformer.pt")
    parser.add_argument("--output", type=str, default="models/char_transformer/quantized_char_transformer.pt")
    args = parser.parse_args()

    export_and_quantize_char_transformer(args.ckpt, args.output)


if __name__ == "__main__":
    main()
