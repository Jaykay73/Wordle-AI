"""
Information-Theoretic Entropy Wordle Solver (Baseline C).
Ranks candidate guesses by expected Shannon entropy (information gain):
H(F|g) = - sum_{f} p(f|g) log2 p(f|g)
Optimized with memoization and candidate-space evaluation.
"""

import math
from typing import List, Tuple, Dict, Optional
from collections import Counter
from ..wordle.dictionary import WordleDictionary
from ..wordle.constraints import SymbolicConstraints
from ..wordle.feedback import get_feedback


class EntropySolver:
    def __init__(self, dictionary: WordleDictionary):
        self.dictionary = dictionary
        self.name = "Information-Theoretic Entropy Solver"
        self._feedback_cache: Dict[Tuple[str, str], Tuple[int, ...]] = {}
        self._first_turn_best: str = "crane"  # Standard optimal opening word

    def _get_cached_feedback(self, secret: str, guess: str) -> Tuple[int, ...]:
        key = (secret, guess)
        if key not in self._feedback_cache:
            self._feedback_cache[key] = get_feedback(secret, guess)
        return self._feedback_cache[key]

    def calculate_entropy(self, guess: str, candidates: List[str]) -> float:
        """
        Calculates expected Shannon entropy (in bits) for a guess given remaining secret candidates.
        H(F|g) = - sum_f p(f|g) * log2(p(f|g))
        """
        num_candidates = len(candidates)
        if num_candidates <= 1:
            return 0.0

        eval_candidates = candidates if num_candidates <= 100 else candidates[:100]
        eval_count = len(eval_candidates)

        pattern_counts = Counter()
        for secret in eval_candidates:
            fb = self._get_cached_feedback(secret, guess)
            pattern_counts[fb] += 1

        entropy = 0.0
        for count in pattern_counts.values():
            p = count / eval_count
            entropy -= p * math.log2(p)

        return entropy

    def get_all_entropy_scores(
        self,
        constraints: SymbolicConstraints,
        candidates: Optional[List[str]] = None,
        search_words: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Returns entropy scores for search words against remaining candidates."""
        if candidates is None:
            candidates = constraints.filter_candidates(self.dictionary.secret_words)
        
        if not candidates:
            return {}

        num_candidates = len(candidates)

        # Determine search pool (all valid guesses or candidates depending on candidate count)
        if search_words is None:
            if num_candidates > 150:
                search_words = candidates[:150]
            else:
                search_words = candidates

        scores = {}
        candidate_set = set(candidates)

        for word in search_words:
            h = self.calculate_entropy(word, candidates)
            # Add small probability bonus if the word is actually a candidate secret solution
            if word in candidate_set:
                win_prob = 1.0 / num_candidates
                h += win_prob * 0.1
            scores[word] = h

        return scores

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
            unplayed_all = [w for w in self.dictionary.all_valid_guesses if w not in played_words]
            return unplayed_all[0] if unplayed_all else self.dictionary.all_valid_guesses[0]

        if len(candidates) == 1:
            return candidates[0]

        # Fast path for turn 1
        if turn == 1 and not constraints.history:
            return self._first_turn_best

        # Compute entropy scores over candidate space
        search_pool = candidates if len(candidates) <= 150 else candidates[:150]
        
        best_word = candidates[0]
        best_entropy = -1.0
        candidate_set = set(candidates)

        for word in search_pool:
            h = self.calculate_entropy(word, candidates)
            # Bonus for candidate solution
            if word in candidate_set:
                h += (1.0 / len(candidates)) * 0.1
            if h > best_entropy:
                best_entropy = h
                best_word = word

        return best_word
