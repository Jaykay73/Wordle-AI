"""
Wordle Core Engine Module.
Provides feedback computation, vocabulary management, symbolic constraints, state serialization, and game mechanics.
"""

from .feedback import get_feedback, feedback_to_emoji, FeedbackTile
from .dictionary import WordleDictionary
from .constraints import SymbolicConstraints
from .state import WordleState
from .game import WordleGame

__all__ = [
    "get_feedback",
    "feedback_to_emoji",
    "FeedbackTile",
    "WordleDictionary",
    "SymbolicConstraints",
    "WordleState",
    "WordleGame",
]
