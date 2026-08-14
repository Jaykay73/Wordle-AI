"""
Letter-Frequency Wordle Solver (Baseline B).
Scores and ranks candidate words based on positional and global character frequency distributions.
"""

from typing import List, Optional, Dict, Tuple
from collections import Counter
from ..wordle.dictionary import WordleDictionary
from ..wordle.constraints import SymbolicConstraints


class FrequencySolver:
    def __init__(self, dictionary: WordleDictionary):
        self.dictionary = dictionary
        self.name = "Letter Frequency Solver"

    def compute_frequencies(self, candidate_list: List[str]) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """Computes normalized global and positional character frequencies across candidate words."""
        total = len(candidate_list)
        if total == 0:
            return {}, [{} for _ in range(5)]

        global_counts = Counter()
        positional_counts = [Counter() for _ in range(5)]

        for word in candidate_list:
            for i, char in enumerate(word):
                global_counts[char] += 1
                positional_counts[i][char] += 1

        global_freq = {c: count / (total * 5) for c, count in global_counts.items()}
        positional_freq = [{c: count / total for c, count in pos_c.items()} for pos_c in positional_counts]

        return global_freq, positional_freq

    def score_word(self, word: str, global_freq: Dict[str, float], positional_freq: List[Dict[str, float]]) -> float:
        """Scores a word favoring unique letters with high positional and global frequencies."""
        seen = set()
        score = 0.0
        for i, char in enumerate(word):
            pos_score = positional_freq[i].get(char, 0.0)
            glob_score = global_freq.get(char, 0.0)
            # Bonus for distinct letters to maximize information gain
            if char not in seen:
                score += pos_score + glob_score
                seen.add(char)
            else:
                score += 0.2 * (pos_score + glob_score)
        return score

    def choose_guess(
        self,
        constraints: SymbolicConstraints,
        turn: int = 1,
        allowed_guesses: Optional[List[str]] = None
    ) -> str:
        candidates = constraints.filter_candidates(self.dictionary.secret_words)
        if not candidates:
            candidates = self.dictionary.all_valid_guesses

        # For turn 1, if precomputed opening word like "CRANE" or "SLATE" is desired or frequency choice:
        global_freq, positional_freq = self.compute_frequencies(candidates)
        
        # Search among candidate secret solutions for optimal guess
        search_space = candidates
        
        best_word = candidates[0]
        best_score = -1.0
        for word in search_space:
            score = self.score_word(word, global_freq, positional_freq)
            # Add small tie-breaker bias for words that are also valid secret candidates
            if word in candidates:
                score += 0.01
            if score > best_score:
                best_score = score
                best_word = word

        return best_word
