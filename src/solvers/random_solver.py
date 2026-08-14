"""
Random Wordle Solver (Baseline A).
Selects a random valid word from the remaining candidate space.
"""

import random
from typing import List, Optional
from ..wordle.dictionary import WordleDictionary
from ..wordle.constraints import SymbolicConstraints


class RandomSolver:
    def __init__(self, dictionary: WordleDictionary, seed: Optional[int] = None):
        self.dictionary = dictionary
        self.rng = random.Random(seed)
        self.name = "Random Solver"

    def choose_guess(
        self,
        constraints: SymbolicConstraints,
        turn: int = 1,
        allowed_guesses: Optional[List[str]] = None
    ) -> str:
        candidates = constraints.filter_candidates(self.dictionary.secret_words)
        if not candidates:
            candidates = self.dictionary.all_valid_guesses
        return self.rng.choice(candidates)
