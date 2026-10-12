"""
Character Transformer Wordle Solver.
Uses a character-level Causal Transformer to rank legal candidate guesses (constrained decoding).
"""

from typing import List, Optional
import os
import torch
from ..wordle.dictionary import WordleDictionary
from ..wordle.constraints import SymbolicConstraints
from ..wordle.state import WordleState
from ..models.char_transformer import WordleCharTransformer, CharTransformerConfig
from ..training.dataset import CharTokenizer
from ..utils.device import get_device


class CharacterTransformerSolver:
    def __init__(
        self,
        dictionary: WordleDictionary,
        model: Optional[WordleCharTransformer] = None,
        checkpoint_path: Optional[str] = None
    ):
        self.dictionary = dictionary
        self.tokenizer = CharTokenizer()
        self.device = get_device()
        self.name = "Character Transformer Solver"

        if model is not None:
            self.model = model.to(self.device)
        elif checkpoint_path and os.path.exists(checkpoint_path):
            try:
                ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                cfg = ckpt.get("config", CharTransformerConfig())
                self.model = WordleCharTransformer(cfg).to(self.device)
                self.model.load_state_dict(ckpt["model_state_dict"])
                self.model.eval()
            except Exception as e:
                print(f"Warning: Could not load checkpoint from {checkpoint_path}: {e}. Creating untrained model.")
                cfg = CharTransformerConfig(vocab_size=self.tokenizer.vocab_size)
                self.model = WordleCharTransformer(cfg).to(self.device)
        else:
            cfg = CharTransformerConfig(vocab_size=self.tokenizer.vocab_size)
            self.model = WordleCharTransformer(cfg).to(self.device)

    def choose_guess(
        self,
        constraints: SymbolicConstraints,
        turn: int = 1,
        allowed_guesses: Optional[List[str]] = None
    ) -> str:
        candidates = constraints.filter_candidates(self.dictionary.secret_words)
        if not candidates:
            candidates = self.dictionary.all_valid_guesses

        if len(candidates) == 1:
            return candidates[0]

        # Fast path turn 1
        if turn == 1 and not constraints.history:
            return "crane"

        # Cap candidates to top 100 for fast CPU evaluation
        eval_cands = candidates[:100] if len(candidates) > 100 else candidates

        # Construct game state sequence prompt
        history = list(constraints.history)
        state = WordleState(self.dictionary, history=history)
        prompt_seq = state.to_char_seq()

        # Constrained decoding: Score legal candidate words
        scores = self.model.score_candidates(prompt_seq, eval_cands, self.tokenizer, self.device)
        
        best_word = max(scores.items(), key=lambda x: x[1])[0]
        return best_word
