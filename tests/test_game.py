"""
Unit tests for Wordle Game engine lifecycle and turn management.
"""

import pytest
from src.wordle.game import WordleGame, GameStatus


def test_game_win():
    game = WordleGame(secret_word="crane")
    assert game.status == GameStatus.IN_PROGRESS

    fb1 = game.make_guess("slate")
    assert game.turns_used == 1
    assert game.status == GameStatus.IN_PROGRESS

    fb2 = game.make_guess("crane")
    assert game.turns_used == 2
    assert game.status == GameStatus.WON


def test_game_loss():
    game = WordleGame(secret_word="crane", max_turns=3)
    game.make_guess("slate")
    game.make_guess("stare")
    game.make_guess("cater")
    assert game.status == GameStatus.LOST
    assert game.turns_used == 3

    with pytest.raises(RuntimeError):
        game.make_guess("apple")
