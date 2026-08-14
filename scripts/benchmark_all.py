"""
Benchmarking Script for Wordle AI.
Evaluates all four solver families on a fixed test set of secret words, computes metrics,
and saves publication plots into reports/figures/.
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
from src.solvers.random_solver import RandomSolver
from src.solvers.frequency_solver import FrequencySolver
from src.solvers.entropy_solver import EntropySolver
from src.solvers.character_transformer_solver import CharacterTransformerSolver
from src.solvers.hybrid_solver import HybridSolver
from src.evaluation.benchmark import BenchmarkRunner
from src.evaluation.visualization import plot_benchmark_summary


def main():
    parser = argparse.ArgumentParser(description="Run Benchmark Evaluation for all Wordle Solvers")
    parser.add_argument("--num_games", type=int, default=50, help="Number of test secret games to benchmark")
    parser.add_argument("--output_dir", type=str, default="reports/figures/", help="Directory to save plot figures")
    args = parser.parse_args()

    dict_mgr = WordleDictionary(seed=42)
    test_secrets = dict_mgr.test_secrets[: args.num_games]

    print(f"==================================================")
    print(f" WORDLE AI BENCHMARKING SUITE")
    print(f" Test Secrets Count: {len(test_secrets)}")
    print(f"==================================================\n")

    runner = BenchmarkRunner(dict_mgr)

    solvers = [
        RandomSolver(dict_mgr, seed=42),
        FrequencySolver(dict_mgr),
        EntropySolver(dict_mgr),
        CharacterTransformerSolver(dict_mgr),
        HybridSolver(dict_mgr),
    ]

    metrics_results = []
    print(f"{'Model Name':<32} | {'Win Rate':<10} | {'Avg Guesses':<12} | {'Avg Time/Guess':<15}", flush=True)
    print("-" * 75, flush=True)

    for solver in solvers:
        m = runner.evaluate_solver(solver, test_secrets=test_secrets, verbose=False)
        metrics_results.append(m)
        print(f"{m.model_name:<32} | {m.win_rate:>8.1f}% | {m.avg_guesses:>10.2f}   | {m.avg_guess_time_ms:>11.2f} ms", flush=True)

    plot_benchmark_summary(metrics_results, output_dir=args.output_dir)
    print(f"\nSaved benchmark visual plots to {args.output_dir}")


if __name__ == "__main__":
    main()
