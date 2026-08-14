"""
Wordle Game Engine.
Manages game lifecycle, guess submission, status state, feedback history, and turn enforcement.
"""

from enum import Enum
from typing import List, Tuple, Optional
from .feedback import get_feedback, feedback_to_emoji, FeedbackTile
from .dictionary import WordleDictionary


class GameStatus(Enum):
    IN_PROGRESS = "IN_PROGRESS"
    WON = "WON"
    LOST = "LOST"


class WordleGame:
    def __init__(
        self,
        secret_word: str,
        dictionary: Optional[WordleDictionary] = None,
        max_turns: int = 6
    ):
        self.secret_word = secret_word.lower()
        if len(self.secret_word) != 5 or not self.secret_word.isalpha():
            raise ValueError(f"Secret word must be a valid 5-letter alphabetic string, got '{secret_word}'")

        self.max_turns = max_turns
        self.dictionary = dictionary
        
        self.guesses: List[str] = []
        self.feedbacks: List[Tuple[int, ...]] = []
        self.status = GameStatus.IN_PROGRESS

    @property
    def turns_used(self) -> int:
        return len(self.guesses)

    @property
    def turns_remaining(self) -> int:
        return self.max_turns - self.turns_used

    def make_guess(self, guess: str) -> Tuple[int, ...]:
        """
        Submits a guess to the game.
        
        Returns:
            Tuple[int, ...] containing numerical feedback (0=gray, 1=yellow, 2=green).
        """
        if self.status != GameStatus.IN_PROGRESS:
            raise RuntimeError(f"Cannot guess in a finished game. Current status: {self.status.value}")

        guess = guess.lower()
        if len(guess) != 5 or not guess.isalpha():
            raise ValueError(f"Guess must be a 5-letter alphabetic word, got '{guess}'")

        if self.dictionary and not self.dictionary.is_valid_guess(guess):
            raise ValueError(f"Guess '{guess}' is not in the allowed word list.")

        feedback = get_feedback(self.secret_word, guess)
        self.guesses.append(guess)
        self.feedbacks.append(feedback)

        # Check win condition (all 2s / green)
        if all(f == FeedbackTile.GREEN for f in feedback):
            self.status = GameStatus.WON
        elif len(self.guesses) >= self.max_turns:
            self.status = GameStatus.LOST

        return feedback

    def display_board(self) -> str:
        """Returns visual ASCII/emoji representation of the board."""
        lines = [f"Wordle Game ({self.turns_used}/{self.max_turns} turns)"]
        for guess, fb in zip(self.guesses, self.feedbacks):
            lines.append(f"{guess.upper()}  {feedback_to_emoji(fb)}")
        if self.status == GameStatus.WON:
            lines.append("🎉 WON!")
        elif self.status == GameStatus.LOST:
            lines.append(f"❌ LOST! Secret was: {self.secret_word.upper()}")
        return "\n".join(lines)
