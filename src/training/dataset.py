"""
PyTorch Dataset and Character Tokenizer for Wordle Trajectories.
Handles character-level sequence tokenization, padding, and tensor collation.
"""

from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from .generation import TrajectorySample


class CharTokenizer:
    def __init__(self):
        self.special_tokens = [
            "<PAD>", "<UNK>", "<BOS>", "<EOS>",
            "<GUESS>", "</GUESS>", "<FEEDBACK>", "</FEEDBACK>",
            "<NEXT>", "</NEXT>"
        ]
        self.letters = [chr(i) for i in range(ord('a'), ord('z') + 1)]
        self.digits = ["0", "1", "2"]

        self.vocab: List[str] = self.special_tokens + self.letters + self.digits
        self.token2id: Dict[str, int] = {t: i for i, t in enumerate(self.vocab)}
        self.id2token: Dict[int, str] = {i: t for i, t in enumerate(self.vocab)}

        self.pad_id = self.token2id["<PAD>"]
        self.unk_id = self.token2id["<UNK>"]
        self.bos_id = self.token2id["<BOS>"]
        self.eos_id = self.token2id["<EOS>"]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode_sequence(self, prompt: str, target: Optional[str] = None) -> List[int]:
        """
        Tokenizes structured text prompt + target word into sequence of token IDs.
        e.g., "<GUESS>crane</GUESS><FEEDBACK>01200</FEEDBACK><NEXT>" + "slate"
        """
        tokens = [self.bos_id]
        
        # Simple tag-aware tokenizer
        i = 0
        n = len(prompt)
        while i < n:
            if prompt[i] == "<":
                end = prompt.find(">", i)
                if end != -1:
                    tag = prompt[i : end + 1]
                    if tag in self.token2id:
                        tokens.append(self.token2id[tag])
                        i = end + 1
                        continue
            char = prompt[i]
            tokens.append(self.token2id.get(char, self.unk_id))
            i += 1

        if target:
            for char in target.lower():
                tokens.append(self.token2id.get(char, self.unk_id))
            tokens.append(self.eos_id)

        return tokens

    def decode_ids(self, ids: List[int]) -> str:
        """Decodes token IDs back into string representation."""
        chars = []
        for idx in ids:
            tok = self.id2token.get(idx, "")
            if tok not in ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]:
                chars.append(tok)
        return "".join(chars)


class WordleTrajectoryDataset(Dataset):
    def __init__(
        self,
        samples: List[TrajectorySample],
        tokenizer: CharTokenizer,
        max_seq_len: int = 256
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        token_ids = self.tokenizer.encode_sequence(sample.prompt_seq, sample.target_word)

        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[: self.max_seq_len]

        # For Causal LM, input_ids = seq[:-1], labels = seq[1:]
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        labels = torch.tensor(token_ids[1:], dtype=torch.long)

        return input_ids, labels


def collate_fn(batch, pad_id: int):
    input_list, label_list = zip(*batch)
    
    padded_inputs = torch.nn.utils.rnn.pad_sequence(input_list, batch_first=True, padding_value=pad_id)
    padded_labels = torch.nn.utils.rnn.pad_sequence(label_list, batch_first=True, padding_value=-100) # -100 ignored in CrossEntropyLoss

    return padded_inputs, padded_labels


def create_dataloaders(
    train_samples: List[TrajectorySample],
    val_samples: List[TrajectorySample],
    tokenizer: CharTokenizer,
    batch_size: int = 64,
    max_seq_len: int = 256
) -> Tuple[DataLoader, DataLoader]:
    train_ds = WordleTrajectoryDataset(train_samples, tokenizer, max_seq_len=max_seq_len)
    val_ds = WordleTrajectoryDataset(val_samples, tokenizer, max_seq_len=max_seq_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id=tokenizer.pad_id)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id=tokenizer.pad_id)
    )

    return train_loader, val_loader
