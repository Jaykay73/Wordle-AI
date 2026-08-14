"""
Fine-tuning script for Tiny Pretrained Language Model using PEFT/LoRA.
Supports Kaggle GPUs and CPU fallback.
"""

import os
from typing import Optional
import torch
from ..wordle.dictionary import WordleDictionary
from ..training.generation import TrajectoryGenerator
from ..models.tiny_wordle_lm import TinyWordleLM, TRANSFORMERS_AVAILABLE
from ..utils.device import get_device


def train_tiny_lm(
    dict_mgr: WordleDictionary,
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2.0e-4,
    num_samples: int = 5000,
    output_dir: str = "models/tiny_lm/",
    seed: int = 42
) -> Optional[TinyWordleLM]:
    """Fine-tunes a tiny pretrained causal LM on synthetic Wordle instruction dataset."""
    if not TRANSFORMERS_AVAILABLE:
        print("Transformers package not installed. Skipping Tiny LM fine-tuning.")
        return None

    device = get_device()
    print(f"Fine-tuning Tiny LM on device: {device}")

    # Initialize model with LoRA
    model_wrapper = TinyWordleLM(model_name_or_path=model_name, use_lora=True, device=device)
    
    # Generate training dataset
    generator = TrajectoryGenerator(dict_mgr, seed=seed)
    train_samples = generator.generate_dataset(dict_mgr.train_secrets, num_samples_target=num_samples)

    # Simple fine-tuning loop over instruction trajectory prompts
    optimizer = torch.optim.AdamW(model_wrapper.model.parameters(), lr=learning_rate)
    os.makedirs(output_dir, exist_ok=True)

    model_wrapper.model.train()
    print(f"Starting LoRA fine-tuning for {epochs} epochs over {len(train_samples)} samples...")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for i in range(0, len(train_samples), batch_size):
            batch = train_samples[i : i + batch_size]
            prompts = [f"{s.prompt_seq} {s.target_word.upper()}" for s in batch]
            
            inputs = model_wrapper.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
            labels = inputs.input_ids.clone()
            labels[labels == model_wrapper.tokenizer.pad_token_id] = -100

            outputs = model_wrapper.model(**inputs, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (len(train_samples) // batch_size + 1)
        print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f}")

    save_path = os.path.join(output_dir, "adapter_model")
    model_wrapper.model.save_pretrained(save_path)
    print(f"Tiny LM LoRA adapters saved to {save_path}")
    return model_wrapper
