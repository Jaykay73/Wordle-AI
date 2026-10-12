"""
Symbolic Constraint Engine for Wordle.
Tracks knowledge constraints (greens, yellows, grays) from game feedback and filters valid candidate words.
"""

from typing import List, Tuple, Set
from .feedback import get_feedback, FeedbackTile


class SymbolicConstraints:
    def __init__(self):
        self.history: List[Tuple[str, Tuple[int, ...]]] = []

    def add_feedback(self, guess: str, feedback: Tuple[int, ...]):
        """Adds a (guess, feedback) observation to the symbolic constraint engine."""
        self.history.append((guess.lower(), feedback))

    def reset(self):
        """Clears constraint history."""
        self.history.clear()

    def is_compatible(self, candidate: str) -> bool:
        """
        Checks if a candidate word is mathematically compatible with all observed feedback.
        A candidate is compatible if guessing `guess` against secret `candidate` yields `feedback`.
        """
        candidate = candidate.lower()
        for guess, feedback in self.history:
            # An already played non-winning guess can never be the solution
            if candidate == guess and not all(f == FeedbackTile.GREEN for f in feedback):
                return False
            if get_feedback(candidate, guess) != feedback:
                return False
        return True

    def filter_candidates(self, candidates: List[str]) -> List[str]:
        """Filters a candidate word list down to only compatible words."""
        if not self.history:
            return list(candidates)
        return [c for c in candidates if self.is_compatible(c)]

    def get_summary(self) -> dict:
        """
        Generates a readable summary of current constraints:
        confirmed positions (greens), present letters (yellows), absent letters (grays).
        """
        greens = ["_"] * 5
        present_chars = set()
        absent_chars = set()

        for guess, feedback in self.history:
            for i, (char, tile) in enumerate(zip(guess, feedback)):
                if tile == FeedbackTile.GREEN:
                    greens[i] = char.upper()
                    present_chars.add(char)
                elif tile == FeedbackTile.YELLOW:
                    present_chars.add(char)

        # Grays that are not green/yellow anywhere
        for guess, feedback in self.history:
            for char, tile in zip(guess, feedback):
                if tile == FeedbackTile.GRAY and char not in present_chars:
                    absent_chars.add(char)

        return {
            "pattern": "".join(greens),
            "present": sorted(list(present_chars)),
            "absent": sorted(list(absent_chars)),
            "turns_played": len(self.history)
        }
