"""
Tiny Pretrained Language Model Wrapper with PEFT/LoRA Fine-Tuning.
Wraps lightweight open-weights causal LMs (SmolLM2-135M, Pythia-140m, Qwen2.5-0.5B, GPT2)
for Wordle candidate sequence generation and scoring.
"""

from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os

# Workaround for Kaggle environments with preinstalled outdated torchao (< 0.16.0)
try:
    import torchao
    ver = getattr(torchao, "__version__", "0.0.0")
    parts = [int(p) for p in ver.split(".")[:2] if p.isdigit()]
    if parts and parts < [0, 16]:
        sys.modules["torchao"] = None
except Exception:
    pass

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class TinyWordleLM(nn.Module):
    def __init__(
        self,
        model_name_or_path: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.use_lora = use_lora
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("HuggingFace transformers and peft packages are required for TinyWordleLM.")

        print(f"Loading pretrained LM base tokenizer & model: {model_name_or_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.float32 if self.device.type == "cpu" else torch.float16
        )

        if use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                bias="none",
                target_modules=["q_proj", "v_proj"] if hasattr(base_model, "q_proj") else None
            )
            self.model = get_peft_model(base_model, peft_config)
        else:
            self.model = base_model

        self.model.to(self.device)
        self.print_trainable_parameters()

    def print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        pct = (trainable / total) * 100.0 if total else 0.0
        print(f"TinyWordleLM Parameters | Total: {total:,} | Trainable: {trainable:,} ({pct:.2f}%)")

    @torch.no_grad()
    def score_candidates(
        self,
        prompt_text: str,
        candidates: List[str]
    ) -> Dict[str, float]:
        """
        Computes log-likelihood log P(word | prompt_text) for each candidate word.
        """
        self.model.eval()
        prompt_inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        prompt_len = prompt_inputs.input_ids.size(1)

        scores = {}
        for word in candidates:
            full_text = f"{prompt_text} {word.upper()}"
            inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
            
            outputs = self.model(**inputs)
            logits = outputs.logits  # (1, seq_len, vocab_size)
            log_probs = F.log_softmax(logits, dim=-1)

            # Sum log probs of the word tokens
            word_score = 0.0
            seq_len = inputs.input_ids.size(1)
            for pos in range(prompt_len - 1, seq_len - 1):
                target_tok = inputs.input_ids[0, pos + 1]
                word_score += log_probs[0, pos, target_tok].item()

            scores[word] = word_score

        return scores
