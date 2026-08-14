"""
Custom Character-Level Causal Transformer for Wordle.
Implemented directly in PyTorch with configurable embedding dimensions, multi-head causal self-attention,
and constrained candidate scoring logic.
"""

import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..training.dataset import CharTokenizer


@dataclass
class CharTransformerConfig:
    vocab_size: int = 39
    d_model: int = 128
    n_head: int = 4
    num_layers: int = 4
    d_ff: int = 256
    dropout: float = 0.1
    max_seq_len: int = 256


class CausalSelfAttention(nn.Module):
    def __init__(self, config: CharTransformerConfig):
        super().__init__()
        assert config.d_model % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.d_model // config.n_head
        self.d_model = config.d_model

        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)).view(
                1, 1, config.max_seq_len, config.max_seq_len
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)

        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)

        # Causal attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.c_proj(y))


class TransformerBlock(nn.Module):
    def __init__(self, config: CharTransformerConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class WordleCharTransformer(nn.Module):
    def __init__(self, config: CharTransformerConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_seq_len, config.d_model))
        self.drop = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight sharing
        self.tok_emb.weight = self.head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_ids.size()
        assert T <= self.config.max_seq_len, f"Sequence length {T} exceeds maximum {self.config.max_seq_len}"

        token_embeddings = self.tok_emb(input_ids)
        position_embeddings = self.pos_emb[:, :T, :]
        x = self.drop(token_embeddings + position_embeddings)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)

        return logits, loss

    @torch.no_grad()
    def score_candidates(
        self,
        prompt_seq: str,
        candidates: List[str],
        tokenizer: CharTokenizer,
        device: torch.device
    ) -> Dict[str, float]:
        """
        Calculates log likelihood log P(word | prompt_seq) for each candidate word in a single batched forward pass.
        """
        if not candidates:
            return {}

        self.eval()
        base_tokens = tokenizer.encode_sequence(prompt_seq)
        base_len = len(base_tokens)

        # Build batched input tensors
        batch_seqs = []
        word_tok_list = []
        for word in candidates:
            w_toks = [tokenizer.token2id.get(c, tokenizer.unk_id) for c in word.lower()]
            word_tok_list.append(w_toks)
            full_seq = base_tokens + w_toks
            if len(full_seq) > self.config.max_seq_len:
                full_seq = full_seq[-self.config.max_seq_len :]
            batch_seqs.append(full_seq[:-1])

        # Batch forward pass
        inp_tensor = torch.tensor(batch_seqs, dtype=torch.long, device=device)
        logits, _ = self.forward(inp_tensor)  # (B, T, V)
        log_probs = F.log_softmax(logits, dim=-1)

        scores = {}
        for idx, (word, w_toks) in enumerate(zip(candidates, word_tok_list)):
            word_score = 0.0
            for pos_idx, char_tok in enumerate(w_toks):
                step_idx = (base_len - 1) + pos_idx
                if step_idx < log_probs.size(1):
                    word_score += log_probs[idx, step_idx, char_tok].item()
            scores[word] = word_score

        return scores
