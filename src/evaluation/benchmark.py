"""
Batch Evaluation Engine for Wordle Solvers.
Runs game simulations over a fixed test set of secret words and logs timing, candidate spaces, and metrics.
"""

import time
from typing import List, Any, Optional
from tqdm import tqdm
from ..wordle.dictionary import WordleDictionary
from ..wordle.game import WordleGame, GameStatus
from ..wordle.constraints import SymbolicConstraints
from .metrics import GameResult, calculate_metrics, SummaryMetrics


class BenchmarkRunner:
    def __init__(self, dictionary: WordleDictionary, max_turns: int = 6):
        self.dictionary = dictionary
        self.max_turns = max_turns

    def evaluate_solver(
        self,
        solver: Any,
        test_secrets: Optional[List[str]] = None,
        verbose: bool = False
    ) -> SummaryMetrics:
        secrets = test_secrets or self.dictionary.test_secrets
        results: List[GameResult] = []

        solver_name = getattr(solver, "name", solver.__class__.__name__)
        pbar = tqdm(secrets, desc=f"Evaluating {solver_name}", disable=not verbose)

        for secret in pbar:
            game = WordleGame(secret_word=secret, dictionary=self.dictionary, max_turns=self.max_turns)
            constraints = SymbolicConstraints()
            candidate_counts = [len(self.dictionary.secret_words)]
            invalid_guesses = 0

            start_time = time.perf_counter()

            while game.status == GameStatus.IN_PROGRESS:
                turn = game.turns_used + 1
                try:
                    guess = solver.choose_guess(
                        constraints=constraints,
                        turn=turn,
                        allowed_guesses=self.dictionary.all_valid_guesses
                    )
                except Exception as e:
                    # Graceful fallback if solver raises exception
                    invalid_guesses += 1
                    valid_cands = constraints.filter_candidates(self.dictionary.secret_words)
                    guess = valid_cands[0] if valid_cands else self.dictionary.all_valid_guesses[0]

                # Validate guess legality
                if not self.dictionary.is_valid_guess(guess):
                    invalid_guesses += 1
                    valid_cands = constraints.filter_candidates(self.dictionary.secret_words)
                    guess = valid_cands[0] if valid_cands else self.dictionary.all_valid_guesses[0]

                fb = game.make_guess(guess)
                constraints.add_feedback(guess, fb)
                rem_cands = len(constraints.filter_candidates(self.dictionary.secret_words))
                candidate_counts.append(rem_cands)

            end_time = time.perf_counter()

            results.append(GameResult(
                secret=secret,
                won=(game.status == GameStatus.WON),
                guesses_taken=game.turns_used,
                guesses=game.guesses,
                feedbacks=game.feedbacks,
                candidate_counts=candidate_counts,
                total_time_sec=(end_time - start_time),
                invalid_guess_count=invalid_guesses
            ))

        return calculate_metrics(solver_name, results)
