"""
Evaluation Framework Module.
Provides benchmarking engines, metrics calculation, publication-quality visualization, and statistical analysis.
"""

from .metrics import calculate_metrics, SummaryMetrics
from .benchmark import BenchmarkRunner
from .visualization import plot_benchmark_summary
from .statistical_tests import compute_bootstrap_ci

__all__ = [
    "calculate_metrics",
    "SummaryMetrics",
    "BenchmarkRunner",
    "plot_benchmark_summary",
    "compute_bootstrap_ci",
]
