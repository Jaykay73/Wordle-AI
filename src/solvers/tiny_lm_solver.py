"""
Tiny Pretrained LM Solver for Wordle.
Scores legal candidate words using fine-tuned causal LM log probabilities (constrained decoding).
"""

from typing import List, Optional
import torch
from ..wordle.dictionary import WordleDictionary
from ..wordle.constraints import SymbolicConstraints
from ..wordle.state import WordleState
from ..models.tiny_wordle_lm import TinyWordleLM, TRANSFORMERS_AVAILABLE


class TinyLMSolver:
    def __init__(
        self,
        dictionary: WordleDictionary,
        model_wrapper: Optional[TinyWordleLM] = None,
        model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    ):
        self.dictionary = dictionary
        self.name = "Tiny Pretrained LM Solver"

        if model_wrapper is not None:
            self.model_wrapper = model_wrapper
        elif TRANSFORMERS_AVAILABLE:
            try:
                self.model_wrapper = TinyWordleLM(model_name_or_path=model_name, use_lora=False)
            except Exception as e:
                print(f"Warning: Could not load LM {model_name}: {e}")
                self.model_wrapper = None
        else:
            self.model_wrapper = None

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

        if not self.model_wrapper:
            # Fallback if LM is unavailable
            return candidates[0]

        history = list(constraints.history)
        state = WordleState(self.dictionary, history=history)
        prompt_text = state.to_instruction_prompt()

        # Score valid candidates with constrained decoding
        scores = self.model_wrapper.score_candidates(prompt_text, candidates)
        best_word = max(scores.items(), key=lambda x: x[1])[0]

        return best_word
