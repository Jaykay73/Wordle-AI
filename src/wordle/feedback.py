"""
Wordle Feedback Engine.
Implements exact Wordle two-pass evaluation rules for gray (0), yellow (1), and green (2) feedback tiles,
with proper repeated-letter tracking and emoji rendering.
"""

from enum import IntEnum
from typing import List, Tuple, Union
from collections import Counter


class FeedbackTile(IntEnum):
    GRAY = 0
    YELLOW = 1
    GREEN = 2


TILE_EMOJI = {
    FeedbackTile.GRAY: "⬛",
    FeedbackTile.YELLOW: "🟨",
    FeedbackTile.GREEN: "🟩",
}


def get_feedback(secret: str, guess: str) -> Tuple[int, ...]:
    """
    Computes exact Wordle feedback for a guess against a secret word.
    
    Args:
        secret: 5-letter secret word (lowercase)
        guess: 5-letter guess word (lowercase)
        
    Returns:
        Tuple[int, ...] of length 5 containing 0 (GRAY), 1 (YELLOW), or 2 (GREEN).
    """
    secret = secret.lower()
    guess = guess.lower()
    
    if len(secret) != 5 or len(guess) != 5:
        raise ValueError(f"Both secret and guess must be 5 characters long. Got secret='{secret}', guess='{guess}'")

    n = len(secret)
    feedback = [FeedbackTile.GRAY] * n
    secret_counts = Counter(secret)

    # Pass 1: Mark exact matches (GREEN)
    for i in range(n):
        if guess[i] == secret[i]:
            feedback[i] = FeedbackTile.GREEN
            secret_counts[guess[i]] -= 1

    # Pass 2: Mark misplaced letters (YELLOW)
    for i in range(n):
        if feedback[i] == FeedbackTile.GRAY:
            g_char = guess[i]
            if secret_counts.get(g_char, 0) > 0:
                feedback[i] = FeedbackTile.YELLOW
                secret_counts[g_char] -= 1

    return tuple(int(x) for x in feedback)


def feedback_to_emoji(feedback: Union[Tuple[int, ...], List[int]]) -> str:
    """Converts numerical feedback tuple/list to visual emoji string (e.g. ⬛🟨🟩⬛🟨)."""
    return "".join(TILE_EMOJI[FeedbackTile(x)] for x in feedback)


def feedback_to_string(feedback: Union[Tuple[int, ...], List[int]]) -> str:
    """Converts numerical feedback tuple to digit string (e.g., '01201')."""
    return "".join(str(int(x)) for x in feedback)


def parse_feedback_string(fb_str: str) -> Tuple[int, ...]:
    """Parses digit string like '01200' into a tuple of ints."""
    return tuple(int(c) for c in fb_str.strip())
