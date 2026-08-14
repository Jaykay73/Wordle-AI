"""
Metrics Calculation for Wordle AI Experiments.
Calculates win rate, guess distributions, candidate reduction curves, latency, and invalid guess ratios.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np


@dataclass
class GameResult:
    secret: str
    won: bool
    guesses_taken: int
    guesses: List[str]
    feedbacks: List[Tuple[int, ...]]
    candidate_counts: List[int]
    total_time_sec: float
    invalid_guess_count: int = 0


@dataclass
class SummaryMetrics:
    model_name: str
    total_games: int
    wins: int
    win_rate: float
    avg_guesses: float
    median_guesses: float
    std_guesses: float
    guess_distribution: Dict[int, int]  # {1: count, 2: count, ..., 6: count, 'fail': count}
    avg_game_time_ms: float
    avg_guess_time_ms: float
    candidate_reduction_curve: List[float]
    invalid_guess_rate: float


def calculate_metrics(model_name: str, results: List[GameResult]) -> SummaryMetrics:
    total_games = len(results)
    if total_games == 0:
        return SummaryMetrics(model_name, 0, 0, 0.0, 0.0, 0.0, 0.0, {}, 0.0, 0.0, [], 0.0)

    wins = sum(1 for r in results if r.won)
    win_rate = (wins / total_games) * 100.0

    won_guesses = [r.guesses_taken for r in results if r.won]
    avg_guesses = float(np.mean(won_guesses)) if won_guesses else 7.0
    median_guesses = float(np.median(won_guesses)) if won_guesses else 7.0
    std_guesses = float(np.std(won_guesses)) if len(won_guesses) > 1 else 0.0

    # Guess distribution
    guess_dist = {i: 0 for i in range(1, 7)}
    guess_dist["fail"] = 0
    for r in results:
        if r.won:
            guess_dist[r.guesses_taken] += 1
        else:
            guess_dist["fail"] += 1

    total_game_time = sum(r.total_time_sec for r in results)
    total_guesses = sum(r.guesses_taken for r in results)
    avg_game_time_ms = (total_game_time / total_games) * 1000.0 if total_games else 0.0
    avg_guess_time_ms = (total_game_time / total_guesses) * 1000.0 if total_guesses else 0.0

    # Average candidate reduction by turn
    max_turns = max((len(r.candidate_counts) for r in results), default=0)
    curve = []
    for turn in range(max_turns):
        turn_counts = [r.candidate_counts[turn] for r in results if turn < len(r.candidate_counts)]
        curve.append(float(np.mean(turn_counts)) if turn_counts else 0.0)

    total_invalids = sum(r.invalid_guess_count for r in results)
    invalid_rate = (total_invalids / total_guesses) * 100.0 if total_guesses else 0.0

    return SummaryMetrics(
        model_name=model_name,
        total_games=total_games,
        wins=wins,
        win_rate=win_rate,
        avg_guesses=avg_guesses,
        median_guesses=median_guesses,
        std_guesses=std_guesses,
        guess_distribution=guess_dist,
        avg_game_time_ms=avg_game_time_ms,
        avg_guess_time_ms=avg_guess_time_ms,
        candidate_reduction_curve=curve,
        invalid_guess_rate=invalid_rate
    )
