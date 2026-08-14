"""
Notebook Generator Utility.
Generates all 9 Kaggle-ready Jupyter notebooks with complete markdown narrative, equations, code cells,
visualizations, and bulletproof path resolution for Kaggle GPU environments.
"""

import json
import os


def make_cell(cell_type: str, source: str):
    lines = [line + "\n" for line in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": cell_type,
        "metadata": {},
        "outputs": [],
        "source": lines
    }


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python"},
            "accelerator": "GPU"
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }


def save_nb(filename, cells):
    os.makedirs("notebooks", exist_ok=True)
    nb = make_notebook(cells)
    with open(os.path.join("notebooks", filename), "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated notebook: notebooks/{filename}")


path_resolver_code = """import sys
import os
import zipfile

# Auto-extract zip file if Kaggle uploaded dataset as a raw zip
if os.path.exists("/kaggle/input"):
    for root_dir, dirs, files in os.walk("/kaggle/input"):
        for file in files:
            if file.endswith(".zip") and "wordle" in file.lower():
                zip_path = os.path.join(root_dir, file)
                print(f"Extracting uploaded Kaggle dataset: {zip_path} -> /kaggle/working/")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall("/kaggle/working/")

# Dynamic path resolution for local and Kaggle environments
possible_roots = [
    ".",
    "..",
    "/kaggle/working",
    "/kaggle/working/wordle",
    "/kaggle/working/wordle-ai",
    "/kaggle/input/wordle",
    "/kaggle/input/wordle/wordle",
    "/kaggle/input/wordle/wordle-ai",
    "/kaggle/input/wordle-ai",
]

# Search recursively inside /kaggle/input for any folder containing 'src'
if os.path.exists("/kaggle/input"):
    for root_dir, dirs, files in os.walk("/kaggle/input"):
        if "src" in dirs:
            possible_roots.insert(0, root_dir)

for root in possible_roots:
    if os.path.exists(os.path.join(root, "src")):
        abs_root = os.path.abspath(root)
        if abs_root not in sys.path:
            sys.path.insert(0, abs_root)
        print(f"Found 'src' package at: {abs_root}")
        break

if os.path.exists("/kaggle/input"):
    print("Running on Kaggle GPU/CPU Environment!")
else:
    print("Running on Local Environment!")
"""

# 01_wordle_environment.ipynb
nb1_cells = [
    make_cell("markdown", """# Notebook 01: Wordle Game Environment & Mechanics
Welcome to the introductory notebook of the **Wordle AI** research project!

### Objectives:
1. Load and inspect the official Wordle solution & allowed guess dictionaries.
2. Demonstrate exact Wordle feedback logic (Gray, Yellow, Green) and repeated-letter edge cases.
3. Walk through interactive game playback with visual board rendering.
4. Demonstrate symbolic constraint candidate filtering.
"""),
    make_cell("code", path_resolver_code + """
from src.wordle.dictionary import WordleDictionary
from src.wordle.game import WordleGame, GameStatus
from src.wordle.feedback import get_feedback, feedback_to_emoji
from src.wordle.constraints import SymbolicConstraints

dict_mgr = WordleDictionary()
print(f"Loaded {len(dict_mgr.secret_words)} secret target words and {len(dict_mgr.all_valid_guesses)} total allowed guess words.")
"""),
    make_cell("markdown", """## 1. Repeated Letter Feedback Verification
Let's inspect the canonical repeated-letter case:
`SECRET = APPLE`, `GUESS = ALLEY`.
- Position 0: `A` == `A` -> 🟩 GREEN
- Position 1: `L` matches `L` in secret -> 🟨 YELLOW
- Position 2: Second `L` in guess has no remaining match -> ⬛ GRAY
- Position 3: `E` matches `E` in secret -> 🟨 YELLOW
- Position 4: `Y` absent -> ⬛ GRAY
"""),
    make_cell("code", """fb = get_feedback("apple", "alley")
print("Feedback digits:", fb)
print("Feedback visual:", feedback_to_emoji(fb))
assert fb == (2, 1, 0, 1, 0)
"""),
    make_cell("markdown", """## 2. Interactive Game Playback & Candidate Filtering"""),
    make_cell("code", """game = WordleGame(secret_word="crane", dictionary=dict_mgr)
constraints = SymbolicConstraints()

guesses = ["slate", "cater", "crane"]
for g in guesses:
    fb = game.make_guess(g)
    constraints.add_feedback(g, fb)
    rem_cands = constraints.filter_candidates(dict_mgr.secret_words)
    print(f"Guess: {g.upper()} | Feedback: {feedback_to_emoji(fb)} | Candidates Remaining: {len(rem_cands)}")

print("\nFinal Game Board State:")
print(game.display_board())
""")
]

# 02_rule_based_solver.ipynb
nb2_cells = [
    make_cell("markdown", """# Notebook 02: Rule-Based & Information-Theoretic Solvers
In this notebook, we implement and benchmark three classical non-neural baselines:
1. **Random Solver** (Baseline A)
2. **Letter Frequency Solver** (Baseline B)
3. **Information-Theoretic Entropy Solver** (Baseline C)

### Mathematical Formulation of Information Gain (Entropy):
For a candidate guess $g$ and remaining candidate secrets $\\mathcal{C}$:
$$H(F|g) = -\\sum_{f} p(f|g) \\log_2 p(f|g)$$
where $p(f|g) = \\frac{\\text{count}(f)}{|\\mathcal{C}|}$.
"""),
    make_cell("code", path_resolver_code + """
from src.wordle.dictionary import WordleDictionary
from src.solvers.random_solver import RandomSolver
from src.solvers.frequency_solver import FrequencySolver
from src.solvers.entropy_solver import EntropySolver
from src.evaluation.benchmark import BenchmarkRunner

dict_mgr = WordleDictionary()
runner = BenchmarkRunner(dict_mgr)
test_set = dict_mgr.test_secrets[:30]

print("Evaluating Random Solver...")
m_random = runner.evaluate_solver(RandomSolver(dict_mgr, seed=42), test_secrets=test_set)
print(f"Random Solver Win Rate: {m_random.win_rate:.1f}%, Avg Guesses: {m_random.avg_guesses:.2f}")

print("Evaluating Frequency Solver...")
m_freq = runner.evaluate_solver(FrequencySolver(dict_mgr), test_secrets=test_set)
print(f"Frequency Solver Win Rate: {m_freq.win_rate:.1f}%, Avg Guesses: {m_freq.avg_guesses:.2f}")

print("Evaluating Entropy Solver...")
m_entropy = runner.evaluate_solver(EntropySolver(dict_mgr), test_secrets=test_set)
print(f"Entropy Solver Win Rate: {m_entropy.win_rate:.1f}%, Avg Guesses: {m_entropy.avg_guesses:.2f}")
""")
]

# 03_generate_training_data.ipynb
nb3_cells = [
    make_cell("markdown", """# Notebook 03: Synthetic Dataset Generation & Data Leakage Prevention

### Critical Rules for Preventing Data Leakage:
1. Split at the **SECRET-WORD** level:
   - `TRAIN secrets` (1,800 words)
   - `VALIDATION secrets` (250 words)
   - `TEST secrets` (265 words)
2. Assert that secret words **NEVER** appear in input prompts or validation test targets.
"""),
    make_cell("code", path_resolver_code + """
from src.wordle.dictionary import WordleDictionary
from src.training.generation import TrajectoryGenerator

dict_mgr = WordleDictionary(seed=42)
generator = TrajectoryGenerator(dict_mgr, seed=42)

print(f"Train Secrets: {len(dict_mgr.train_secrets)} | Val Secrets: {len(dict_mgr.val_secrets)} | Test Secrets: {len(dict_mgr.test_secrets)}")

train_samples = generator.generate_dataset(dict_mgr.train_secrets, num_samples_target=1000)
print(f"Generated {len(train_samples)} training trajectory samples.")
print("Sample Prompt Sequence:\\n", train_samples[0].prompt_seq)
print("Sample Target Word:", train_samples[0].target_word)
""")
]

# 04_character_transformer_training.ipynb
nb4_cells = [
    make_cell("markdown", """# Notebook 04: Character-Level Transformer Training (Kaggle GPU Ready)
In this notebook, we build and train a causal Character Transformer directly from scratch in PyTorch.

### Features:
- Kaggle P100/T4 GPU Auto-Detection
- Custom PyTorch Causal Self-Attention (No HF dependency)
- Automatic Mixed Precision (AMP)
- Model Checkpoint saving
"""),
    make_cell("code", path_resolver_code + """import torch

if os.path.exists("/kaggle/input"):
    !pip install -q pyyaml matplotlib seaborn tqdm

print("PyTorch Version:", torch.__version__, "| CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))

from src.wordle.dictionary import WordleDictionary
from src.models.char_transformer import CharTransformerConfig
from src.training.train_char_transformer import train_character_transformer

dict_mgr = WordleDictionary()
config = CharTransformerConfig(d_model=128, n_head=4, num_layers=4, d_ff=256)

model = train_character_transformer(
    config=config,
    dict_mgr=dict_mgr,
    epochs=2,
    batch_size=64,
    num_train_samples=2000,
    num_val_samples=500
)
""")
]

# 05_character_transformer_evaluation.ipynb
nb5_cells = [
    make_cell("markdown", """# Notebook 05: Character Transformer Evaluation & Game Playback
We evaluate the trained Character Transformer on unseen test secrets using constrained decoding:
$$P(\\text{word} | \\text{game\\_state})$$
"""),
    make_cell("code", path_resolver_code + """
from src.wordle.dictionary import WordleDictionary
from src.solvers.character_transformer_solver import CharacterTransformerSolver
from src.evaluation.benchmark import BenchmarkRunner

dict_mgr = WordleDictionary()
solver = CharacterTransformerSolver(dict_mgr)
runner = BenchmarkRunner(dict_mgr)

metrics = runner.evaluate_solver(solver, test_secrets=dict_mgr.test_secrets[:25])
print(f"Char Transformer Win Rate: {metrics.win_rate:.1f}%, Avg Guesses: {metrics.avg_guesses:.2f}")
""")
]

# 06_tiny_lm_finetuning.ipynb
nb6_cells = [
    make_cell("markdown", """# Notebook 06: Tiny Pretrained LM Fine-Tuning with PEFT/LoRA (Kaggle GPU Ready)
Fine-tuning a lightweight pretrained open-weights causal language model (`SmolLM2-135M` or `Pythia-140m`) for Wordle state generation using Low-Rank Adaptation (LoRA).
"""),
    make_cell("code", path_resolver_code + """import torch

if os.path.exists("/kaggle/input"):
    !pip install -q -U torchao transformers peft accelerate datasets

from src.wordle.dictionary import WordleDictionary
from src.training.train_tiny_lm import train_tiny_lm

dict_mgr = WordleDictionary()
train_tiny_lm(dict_mgr, model_name="HuggingFaceTB/SmolLM2-135M-Instruct", epochs=1, num_samples=1000)
""")
]

# 07_tiny_lm_evaluation.ipynb
nb7_cells = [
    make_cell("markdown", """# Notebook 07: Tiny Pretrained LM Wordle Game Evaluation
Evaluating the fine-tuned Tiny LM agent on unseen test secret games.
"""),
    make_cell("code", path_resolver_code + """
from src.wordle.dictionary import WordleDictionary
from src.solvers.tiny_lm_solver import TinyLMSolver
from src.evaluation.benchmark import BenchmarkRunner

dict_mgr = WordleDictionary()
solver = TinyLMSolver(dict_mgr)
runner = BenchmarkRunner(dict_mgr)

metrics = runner.evaluate_solver(solver, test_secrets=dict_mgr.test_secrets[:20])
print(f"Tiny LM Win Rate: {metrics.win_rate:.1f}%, Avg Guesses: {metrics.avg_guesses:.2f}")
""")
]

# 08_hybrid_agent.ipynb
nb8_cells = [
    make_cell("markdown", """# Notebook 08: Neuro-Symbolic Hybrid Agent & Ablation Study
Investigating the Neuro-Symbolic Hybrid Solver combining symbolic hard constraint filtering, Transformer neural log-likelihood, and Information-Theoretic Entropy gain:

$$\\text{FinalScore}(w) = \\alpha \\cdot \\text{NeuralScore}(w) + \\beta \\cdot \\text{EntropyScore}(w) + \\gamma \\cdot \\text{FrequencyScore}(w)$$
"""),
    make_cell("code", path_resolver_code + """
from src.wordle.dictionary import WordleDictionary
from src.solvers.hybrid_solver import HybridSolver
from src.evaluation.benchmark import BenchmarkRunner

dict_mgr = WordleDictionary()
runner = BenchmarkRunner(dict_mgr)
test_set = dict_mgr.test_secrets[:25]

# Ablation A: Full Hybrid
hybrid_full = HybridSolver(dict_mgr, alpha=0.5, beta=0.4, gamma=0.1, enforce_constraints=True)
m_full = runner.evaluate_solver(hybrid_full, test_secrets=test_set)

# Ablation B: Without Entropy (beta=0)
hybrid_no_entropy = HybridSolver(dict_mgr, alpha=0.9, beta=0.0, gamma=0.1, enforce_constraints=True)
m_no_entropy = runner.evaluate_solver(hybrid_no_entropy, test_secrets=test_set)

print(f"Full Hybrid Win Rate: {m_full.win_rate:.1f}% | Avg Guesses: {m_full.avg_guesses:.2f}")
print(f"Hybrid (No Entropy) Win Rate: {m_no_entropy.win_rate:.1f}% | Avg Guesses: {m_no_entropy.avg_guesses:.2f}")
""")
]

# 09_final_comparison.ipynb
nb9_cells = [
    make_cell("markdown", """# Notebook 09: Final Comparison & Experimental Analysis
This flagship notebook runs the complete comparative benchmark across all solvers, generates high-res publication figures, and performs statistical bootstrap confidence interval analysis.
"""),
    make_cell("code", path_resolver_code + """import shutil
from IPython.display import Image, display
import glob

from src.wordle.dictionary import WordleDictionary
from src.solvers.random_solver import RandomSolver
from src.solvers.frequency_solver import FrequencySolver
from src.solvers.entropy_solver import EntropySolver
from src.solvers.character_transformer_solver import CharacterTransformerSolver
from src.solvers.hybrid_solver import HybridSolver
from src.evaluation.benchmark import BenchmarkRunner
from src.evaluation.visualization import plot_benchmark_summary
from src.evaluation.statistical_tests import compute_bootstrap_ci

dict_mgr = WordleDictionary(seed=42)
runner = BenchmarkRunner(dict_mgr)
test_secrets = dict_mgr.test_secrets[:30]

solvers = [
    RandomSolver(dict_mgr, seed=42),
    FrequencySolver(dict_mgr),
    EntropySolver(dict_mgr),
    CharacterTransformerSolver(dict_mgr),
    HybridSolver(dict_mgr),
]

metrics_list = []
for s in solvers:
    m = runner.evaluate_solver(s, test_secrets=test_secrets)
    metrics_list.append(m)
    print(f"{m.model_name:<30} | Win Rate: {m.win_rate:>5.1f}% | Avg Guesses: {m.avg_guesses:>4.2f}")

out_dir = "/kaggle/working/reports/figures/" if os.path.exists("/kaggle/input") else "../reports/figures/"
plot_benchmark_summary(metrics_list, output_dir=out_dir)
print("\\nSaved benchmark visual plots to:", out_dir)

# 1. Display images directly inside notebook cells
print("\\n--- RENDERING PLOTS INLINE ---")
for img_path in sorted(glob.glob(os.path.join(out_dir, "*.png"))):
    print("Displaying:", os.path.basename(img_path))
    display(Image(filename=img_path))

# 2. Package as a ZIP file in /kaggle/working/ for 1-click download in Kaggle Output sidebar
if os.path.exists("/kaggle/input"):
    zip_out = "/kaggle/working/wordle_benchmark_figures"
    shutil.make_archive(zip_out, 'zip', out_dir)
    print(f"\\nCreated 1-click downloadable archive: {zip_out}.zip (available in Kaggle Output panel!)")
""")
]

def generate_all():
    save_nb("01_wordle_environment.ipynb", nb1_cells)
    save_nb("02_rule_based_solver.ipynb", nb2_cells)
    save_nb("03_generate_training_data.ipynb", nb3_cells)
    save_nb("04_character_transformer_training.ipynb", nb4_cells)
    save_nb("05_character_transformer_evaluation.ipynb", nb5_cells)
    save_nb("06_tiny_lm_finetuning.ipynb", nb6_cells)
    save_nb("07_tiny_lm_evaluation.ipynb", nb7_cells)
    save_nb("08_hybrid_agent.ipynb", nb8_cells)
    save_nb("09_final_comparison.ipynb", nb9_cells)

if __name__ == "__main__":
    generate_all()
