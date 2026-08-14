"""
Unit tests for Wordle Feedback computation and repeated letter edge cases.
"""

import pytest
from src.wordle.feedback import get_feedback, feedback_to_emoji, FeedbackTile


def test_basic_feedback():
    # Exact match
    assert get_feedback("crane", "crane") == (2, 2, 2, 2, 2)
    # All absent
    assert get_feedback("crane", "mould") == (0, 0, 0, 0, 0)
    # All yellow
    assert get_feedback("crane", "ranec") == (1, 1, 1, 1, 1)


def test_repeated_letters_apple_alley():
    # SECRET = APPLE (A:1, P:2, L:1, E:1)
    # GUESS = ALLEY
    # Pos 0: A == A -> GREEN (2), remaining A:0
    # Pos 1: L != P -> check remaining L:1 -> YELLOW (1), remaining L:0
    # Pos 2: L != P -> remaining L:0 -> GRAY (0)
    # Pos 3: E != L -> remaining E:1 -> YELLOW (1), remaining E:0
    # Pos 4: Y != E -> remaining Y:0 -> GRAY (0)
    feedback = get_feedback("apple", "alley")
    assert feedback == (2, 1, 0, 1, 0)
    assert feedback_to_emoji(feedback) == "🟩🟨⬛🟨⬛"


def test_repeated_letters_eerie():
    # SECRET = EERIE (E:3, R:1, I:1)
    # GUESS = EERIE -> exact match (2,2,2,2,2)
    assert get_feedback("eerie", "eerie") == (2, 2, 2, 2, 2)
    # GUESS = EEEEE -> Pos 0,1,4 GREEN, Pos 2,3 GRAY because E count is 3 total
    assert get_feedback("eerie", "eeeee") == (2, 2, 0, 0, 2)


def test_repeated_letters_speed_erase():
    # SECRET = SPEED (S:1, P:1, E:2, D:1)
    # GUESS = ERASE (E:2, R:1, A:1, S:1)
    # Pass 1: no exact match
    # Pass 2:
    # Pos 0 E -> YELLOW (1) (E remaining:1)
    # Pos 1 R -> GRAY (0)
    # Pos 2 A -> GRAY (0)
    # Pos 3 S -> YELLOW (1) (S remaining:0)
    # Pos 4 E -> YELLOW (1) (E remaining:0)
    assert get_feedback("speed", "erase") == (1, 0, 0, 1, 1)


def test_feedback_case_insensitive():
    assert get_feedback("CRANE", "crane") == (2, 2, 2, 2, 2)
    assert get_feedback("crane", "CRANE") == (2, 2, 2, 2, 2)
