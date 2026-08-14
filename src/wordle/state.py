"""
Wordle State Serialization & Tracking.
Provides serialization formats for neural models (Character Transformer & Tiny LM)
and tracks current candidate spaces.
"""

from typing import List, Tuple, Optional
from .feedback import feedback_to_string
from .constraints import SymbolicConstraints
from .dictionary import WordleDictionary


class WordleState:
    def __init__(
        self,
        dictionary: WordleDictionary,
        history: Optional[List[Tuple[str, Tuple[int, ...]]]] = None
    ):
        self.dictionary = dictionary
        self.history: List[Tuple[str, Tuple[int, ...]]] = history or []
        self.constraints = SymbolicConstraints()
        for guess, fb in self.history:
            self.constraints.add_feedback(guess, fb)

    def add_turn(self, guess: str, feedback: Tuple[int, ...]):
        self.history.append((guess.lower(), feedback))
        self.constraints.add_feedback(guess, feedback)

    def get_candidate_words(self) -> List[str]:
        """Returns valid candidate words filtered by current symbolic constraints."""
        return self.constraints.filter_candidates(self.dictionary.secret_words)

    def to_char_seq(self) -> str:
        """
        Serializes game state into structured character token format:
        <GUESS>crane</GUESS><FEEDBACK>01200</FEEDBACK><GUESS>slate</GUESS><FEEDBACK>00120</FEEDBACK><NEXT>
        """
        seq_parts = []
        for guess, fb in self.history:
            fb_str = feedback_to_string(fb)
            seq_parts.append(f"<GUESS>{guess.lower()}</GUESS><FEEDBACK>{fb_str}</FEEDBACK>")
        seq_parts.append("<NEXT>")
        return "".join(seq_parts)

    def to_instruction_prompt(self) -> str:
        """
        Serializes game state into instruction-style prompt format for Causal LMs:
        Wordle game state:
        Guess 1: CRANE | Feedback: 01200
        Guess 2: SLATE | Feedback: 00120
        Next guess: <NEXT_WORD>
        """
        if not self.history:
            return "Wordle game state:\nNext guess: <NEXT_WORD>"

        lines = ["Wordle game state:"]
        for i, (guess, fb) in enumerate(self.history, start=1):
            fb_str = feedback_to_string(fb)
            lines.append(f"Guess {i}: {guess.upper()} | Feedback: {fb_str}")
        lines.append("Next guess: <NEXT_WORD>")
        return "\n".join(lines)
