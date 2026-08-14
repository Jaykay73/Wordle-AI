"""
Visualization Suite for Wordle AI Benchmark Results.
Generates publication-quality charts for win rates, guess distributions, candidate reduction curves, and latency.
"""

import os
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .metrics import SummaryMetrics

# Set clean visual style
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})


def plot_benchmark_summary(
    metrics_list: List[SummaryMetrics],
    output_dir: str = "reports/figures/"
):
    """Generates and saves all benchmark plots into output_dir."""
    if os.path.exists("/kaggle/input") and not output_dir.startswith("/kaggle/working"):
        output_dir = "/kaggle/working/reports/figures/"

    os.makedirs(output_dir, exist_ok=True)

    models = [m.model_name for m in metrics_list]
    win_rates = [m.win_rate for m in metrics_list]
    avg_guesses = [m.avg_guesses for m in metrics_list]
    latencies = [m.avg_guess_time_ms for m in metrics_list]

    # 1. Win Rate Bar Chart
    plt.figure(figsize=(9, 5))
    bars = plt.bar(models, win_rates, color=sns.color_palette("mako", len(models)))
    plt.ylabel("Win Rate (%)")
    plt.title("Wordle Solvers — Win Rate Comparison", fontsize=14, fontweight="bold")
    plt.ylim(0, 105)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1.5, f"{height:.1f}%", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "win_rate_comparison.png"), dpi=300)
    plt.close()

    # 2. Average Guesses Comparison
    plt.figure(figsize=(9, 5))
    bars = plt.bar(models, avg_guesses, color=sns.color_palette("viridis", len(models)))
    plt.ylabel("Average Guesses (Lower is Better)")
    plt.title("Wordle Solvers — Average Guesses to Win", fontsize=14, fontweight="bold")
    plt.ylim(0, 7)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1, f"{height:.2f}", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "avg_guesses_comparison.png"), dpi=300)
    plt.close()

    # 3. Guess Distribution Stacked/Grouped Histogram
    plt.figure(figsize=(10, 6))
    turns = [1, 2, 3, 4, 5, 6, "fail"]
    x = np.arange(len(turns))
    width = 0.8 / len(metrics_list)

    for i, m in enumerate(metrics_list):
        counts = [m.guess_distribution.get(t, 0) for t in turns]
        pcts = [(c / m.total_games)*100 if m.total_games else 0 for c in counts]
        plt.bar(x + i * width, pcts, width=width, label=m.model_name)

    plt.xlabel("Turn Count / Outcome")
    plt.ylabel("Percentage of Games (%)")
    plt.title("Guess Distribution Across Solvers", fontsize=14, fontweight="bold")
    plt.xticks(x + width * (len(metrics_list) - 1) / 2, ["Turn 1", "Turn 2", "Turn 3", "Turn 4", "Turn 5", "Turn 6", "Fail"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "guess_distribution.png"), dpi=300)
    plt.close()

    # 4. Candidate Space Reduction Curve
    plt.figure(figsize=(9, 5))
    for m in metrics_list:
        if m.candidate_reduction_curve:
            turns_idx = list(range(len(m.candidate_reduction_curve)))
            plt.plot(turns_idx, m.candidate_reduction_curve, marker="o", label=m.model_name, linewidth=2)
    plt.yscale("log")
    plt.xlabel("Turn Index")
    plt.ylabel("Remaining Candidates (Log Scale)")
    plt.title("Candidate Space Reduction by Turn", fontsize=14, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "candidate_reduction_curve.png"), dpi=300)
    plt.close()

    # 5. Latency vs Win Rate Scatter Plot
    plt.figure(figsize=(8, 5))
    for m in metrics_list:
        plt.scatter(m.avg_guess_time_ms, m.win_rate, s=120, label=m.model_name)
        plt.annotate(m.model_name, (m.avg_guess_time_ms, m.win_rate + 0.8), fontsize=9)
    plt.xlabel("Inference Latency per Guess (ms)")
    plt.ylabel("Win Rate (%)")
    plt.title("Win Rate vs. Inference Latency Tradeoff", fontsize=14, fontweight="bold")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latency_vs_accuracy.png"), dpi=300)
    plt.close()
