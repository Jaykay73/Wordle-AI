"""
Unit tests for Wordle Solvers (Random, Frequency, Entropy).
"""

from src.wordle.dictionary import WordleDictionary
from src.wordle.constraints import SymbolicConstraints
from src.wordle.game import WordleGame, GameStatus
from src.solvers.random_solver import RandomSolver
from src.solvers.frequency_solver import FrequencySolver
from src.solvers.entropy_solver import EntropySolver


def test_random_solver():
    dict_mgr = WordleDictionary()
    solver = RandomSolver(dict_mgr, seed=42)
    game = WordleGame(secret_word="crane", dictionary=dict_mgr)
    constraints = SymbolicConstraints()

    for turn in range(1, 7):
        guess = solver.choose_guess(constraints, turn=turn)
        assert dict_mgr.is_valid_guess(guess)
        fb = game.make_guess(guess)
        constraints.add_feedback(guess, fb)
        if game.status != GameStatus.IN_PROGRESS:
            break

    assert game.turns_used > 0


def test_frequency_solver():
    dict_mgr = WordleDictionary()
    solver = FrequencySolver(dict_mgr)
    constraints = SymbolicConstraints()
    guess = solver.choose_guess(constraints, turn=1)
    assert dict_mgr.is_valid_guess(guess)


def test_entropy_solver():
    dict_mgr = WordleDictionary()
    solver = EntropySolver(dict_mgr)
    constraints = SymbolicConstraints()

    guess1 = solver.choose_guess(constraints, turn=1)
    assert guess1 == "crane"

    # Add feedback for guess1 against secret "slate"
    constraints.add_feedback("crane", (0, 0, 2, 1, 1))
    guess2 = solver.choose_guess(constraints, turn=2)
    assert dict_mgr.is_valid_guess(guess2)
