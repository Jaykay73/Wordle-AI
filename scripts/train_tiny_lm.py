"""
CLI script to fine-tune Tiny Pretrained LM with PEFT/LoRA.
"""

import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.wordle.dictionary import WordleDictionary
from src.training.train_tiny_lm import train_tiny_lm


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Tiny Pretrained LM for Wordle")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct")

    args = parser.parse_args()

    dict_mgr = WordleDictionary()
    train_tiny_lm(
        dict_mgr=dict_mgr,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_samples=2000
    )


if __name__ == "__main__":
    main()
