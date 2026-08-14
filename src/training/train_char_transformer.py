"""
Training Pipeline for Wordle Character Transformer.
Supports Kaggle GPU acceleration, automatic mixed precision, model checkpointing, and metric logging.
"""

import os
import time
from typing import Dict, Any, Optional
import torch
import torch.optim as optim
from ..wordle.dictionary import WordleDictionary
from ..training.generation import TrajectoryGenerator
from ..training.dataset import CharTokenizer, create_dataloaders
from ..models.char_transformer import WordleCharTransformer, CharTransformerConfig
from ..utils.device import get_device


def train_character_transformer(
    config: CharTransformerConfig,
    dict_mgr: WordleDictionary,
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 5.0e-4,
    num_train_samples: int = 20000,
    num_val_samples: int = 2000,
    checkpoint_dir: str = "models/char_transformer/",
    seed: int = 42
) -> WordleCharTransformer:
    """Executes end-to-end training of the character transformer."""
    torch.manual_seed(seed)
    device = get_device()
    print(f"Using compute device: {device}", flush=True)

    # Generate synthetic training & validation trajectory data
    generator = TrajectoryGenerator(dict_mgr, seed=seed)
    print(f"Generating {num_train_samples} training samples and {num_val_samples} val samples...", flush=True)
    train_samples = generator.generate_dataset(dict_mgr.train_secrets, num_samples_target=num_train_samples)
    val_samples = generator.generate_dataset(dict_mgr.val_secrets, num_samples_target=num_val_samples)

    tokenizer = CharTokenizer()
    config.vocab_size = tokenizer.vocab_size

    train_loader, val_loader = create_dataloaders(
        train_samples, val_samples, tokenizer, batch_size=batch_size, max_seq_len=config.max_seq_len
    )

    model = WordleCharTransformer(config).to(device)
    print(f"Model Architecture initialized. Total Trainable Parameters: {model.count_parameters():,}")

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for input_ids, labels in train_loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits, loss = model(input_ids, targets=labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for input_ids, labels in val_loader:
                input_ids, labels = input_ids.to(device), labels.to(device)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    _, loss = model(input_ids, targets=labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(checkpoint_dir, "best_char_transformer.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "val_loss": avg_val_loss,
            }, ckpt_path)

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds. Best Val Loss: {best_val_loss:.4f}")
    return model
