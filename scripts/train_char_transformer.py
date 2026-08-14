"""
CLI script to train Character Transformer.
"""

import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.wordle.dictionary import WordleDictionary
from src.models.char_transformer import CharTransformerConfig
from src.training.train_char_transformer import train_character_transformer


def main():
    parser = argparse.ArgumentParser(description="Train Character-Level Wordle Transformer")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--num_samples", type=int, default=5000, help="Number of synthetic samples")

    args = parser.parse_args()

    dict_mgr = WordleDictionary()
    config = CharTransformerConfig()

    train_character_transformer(
        config=config,
        dict_mgr=dict_mgr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_samples=args.num_samples,
        num_val_samples=1000
    )


if __name__ == "__main__":
    main()
