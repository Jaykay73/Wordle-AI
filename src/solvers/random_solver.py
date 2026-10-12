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
            candidates = constraints.filter_candidates(self.dictionary.all_valid_guesses)

        played_words = {guess.lower() for guess, _ in constraints.history}
        unplayed = [c for c in candidates if c not in played_words]
        if unplayed:
            candidates = unplayed

        if not candidates:
            candidates = [w for w in self.dictionary.all_valid_guesses if w not in played_words]

        return self.rng.choice(candidates or self.dictionary.all_valid_guesses)
