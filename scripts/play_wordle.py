"""
Interactive CLI for Wordle AI Agent.
Allows interactive game playback or human assistant mode (user inputs external Wordle feedback).
"""

import sys
import os
import argparse

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.wordle.dictionary import WordleDictionary
from src.wordle.game import WordleGame, GameStatus
from src.wordle.constraints import SymbolicConstraints
from src.wordle.feedback import feedback_to_emoji, parse_feedback_string
from src.solvers.random_solver import RandomSolver
from src.solvers.frequency_solver import FrequencySolver
from src.solvers.entropy_solver import EntropySolver
from src.solvers.character_transformer_solver import CharacterTransformerSolver
from src.solvers.tiny_lm_solver import TinyLMSolver
from src.solvers.hybrid_solver import HybridSolver


def get_solver(choice: str, dict_mgr: WordleDictionary):
    if choice == "1" or choice == "random":
        return RandomSolver(dict_mgr)
    elif choice == "2" or choice == "frequency":
        return FrequencySolver(dict_mgr)
    elif choice == "3" or choice == "entropy":
        return EntropySolver(dict_mgr)
    elif choice == "4" or choice == "char_transformer":
        return CharacterTransformerSolver(dict_mgr)
    elif choice == "5" or choice == "tiny_lm":
        return TinyLMSolver(dict_mgr)
    else:
        return HybridSolver(dict_mgr)


def play_auto_game(dict_mgr: WordleDictionary, solver, secret: str = "crane"):
    print(f"\n========================================")
    print(f" WORDLE AI PLAYBOOK — {solver.name.upper()}")
    print(f" Target Secret Word: {secret.upper()}")
    print(f"========================================\n")

    game = WordleGame(secret_word=secret, dictionary=dict_mgr)
    constraints = SymbolicConstraints()

    while game.status == GameStatus.IN_PROGRESS:
        turn = game.turns_used + 1
        guess = solver.choose_guess(constraints, turn=turn)
        fb = game.make_guess(guess)
        constraints.add_feedback(guess, fb)
        rem_cands = len(constraints.filter_candidates(dict_mgr.secret_words))

        print(f"Turn {turn} | AI Guess: {guess.upper()}  {feedback_to_emoji(fb)} | Candidates remaining: {rem_cands}")

    if game.status == GameStatus.WON:
        print(f"\n🎉 SOLVED in {game.turns_used} guesses!\n")
    else:
        print(f"\n❌ FAILED to solve in 6 turns. Secret was {secret.upper()}\n")


def human_assistant_mode(dict_mgr: WordleDictionary, solver):
    print(f"\n========================================")
    print(f" HUMAN ASSISTANT MODE — {solver.name.upper()}")
    print(f" Use this mode while playing an actual Wordle game!")
    print(f" Feedback format: 0=GRAY, 1=YELLOW, 2=GREEN (e.g. 01200)")
    print(f"========================================\n")

    constraints = SymbolicConstraints()
    turn = 1

    while turn <= 6:
        cands = constraints.filter_candidates(dict_mgr.secret_words)
        rem = len(cands)
        print(f"--- Turn {turn} (Candidates remaining: {rem}) ---")

        rec_guess = solver.choose_guess(constraints, turn=turn)
        print(f"💡 Recommended AI Guess: {rec_guess.upper()}")

        user_input = input("Enter feedback (5 digits 0/1/2) or 'q' to quit: ").strip()
        if user_input.lower() == "q":
            break

        if len(user_input) != 5 or not user_input.isdigit():
            print("Invalid feedback string. Please type 5 digits like 01200.")
            continue

        fb = parse_feedback_string(user_input)
        print(f"Recorded feedback: {feedback_to_emoji(fb)}")
        constraints.add_feedback(rec_guess, fb)

        if all(tile == 2 for tile in fb):
            print("\n🎉 CONGRATULATIONS! Wordle Solved!\n")
            break

        turn += 1


def main():
    parser = argparse.ArgumentParser(description="Interactive Wordle AI")
    parser.add_argument("--model", type=str, default="hybrid", help="Solver model (random, frequency, entropy, char_transformer, tiny_lm, hybrid)")
    parser.add_argument("--assistant", action="store_true", help="Launch human assistant mode")
    parser.add_argument("--secret", type=str, default="apple", help="Secret word for auto game")

    args = parser.parse_args()
    dict_mgr = WordleDictionary()
    solver = get_solver(args.model, dict_mgr)

    if args.assistant:
        human_assistant_mode(dict_mgr, solver)
    else:
        play_auto_game(dict_mgr, solver, secret=args.secret)


if __name__ == "__main__":
    main()
