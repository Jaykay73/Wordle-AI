"""
Unit tests for Symbolic Constraints filtering.
"""

from src.wordle.constraints import SymbolicConstraints
from src.wordle.feedback import get_feedback


def test_symbolic_constraints():
    constraints = SymbolicConstraints()
    candidates = ["crane", "slate", "stare", "apple", "cater"]

    # Guess "crane" against secret "slate" -> feedback: (0, 0, 2, 1, 1)
    fb = get_feedback("slate", "crane")
    constraints.add_feedback("crane", fb)

    filtered = constraints.filter_candidates(candidates)
    assert "slate" in filtered
    assert "apple" not in filtered
    assert "crane" not in filtered


def test_empty_constraints():
    constraints = SymbolicConstraints()
    candidates = ["crane", "slate"]
    assert constraints.filter_candidates(candidates) == candidates
