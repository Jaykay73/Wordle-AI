"""
Neuro-Symbolic Hybrid Wordle Solver.
Combines symbolic hard constraint filtering with neural prior scores, information-theoretic entropy gain,
and letter frequency rankings:
FinalScore(w) = alpha * NeuralScore(w) + beta * EntropyScore(w) + gamma * FrequencyScore(w)
"""

from typing import List, Optional, Dict, Any
from ..wordle.dictionary import WordleDictionary
from ..wordle.constraints import SymbolicConstraints
from ..wordle.state import WordleState
from ..solvers.entropy_solver import EntropySolver
from ..solvers.frequency_solver import FrequencySolver
from ..solvers.character_transformer_solver import CharacterTransformerSolver


class HybridSolver:
    def __init__(
        self,
        dictionary: WordleDictionary,
        char_transformer_solver: Optional[CharacterTransformerSolver] = None,
        alpha: float = 0.5,   # Neural weight
        beta: float = 0.4,    # Information Gain weight
        gamma: float = 0.1,   # Frequency weight
        enforce_constraints: bool = True
    ):
        self.dictionary = dictionary
        self.neural_solver = char_transformer_solver or CharacterTransformerSolver(dictionary)
        self.entropy_solver = EntropySolver(dictionary)
        self.freq_solver = FrequencySolver(dictionary)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.enforce_constraints = enforce_constraints
        self.name = "Neuro-Symbolic Hybrid Solver"

    def choose_guess(
        self,
        constraints: SymbolicConstraints,
        turn: int = 1,
        allowed_guesses: Optional[List[str]] = None
    ) -> str:
        # Step 1: Symbolic Constraint Filtering (Hard Layer)
        if self.enforce_constraints:
            candidates = constraints.filter_candidates(self.dictionary.secret_words)
            # Fallback: if target secret is not in secret_words (e.g. valid guess like ADIEU), filter all valid guesses
            if not candidates:
                candidates = constraints.filter_candidates(self.dictionary.all_valid_guesses)
        else:
            candidates = constraints.filter_candidates(self.dictionary.all_valid_guesses)

        # Exclude already played guesses from candidate pool
        played_words = {guess.lower() for guess, _ in constraints.history}
        unplayed = [c for c in candidates if c not in played_words]
        if unplayed:
            candidates = unplayed

        if not candidates:
            # Absolute safety fallback: pick unplayed word from all valid guesses
            unplayed_all = [w for w in self.dictionary.all_valid_guesses if w not in played_words]
            return unplayed_all[0] if unplayed_all else self.dictionary.all_valid_guesses[0]

        if len(candidates) == 1:
            return candidates[0]

        # Fast path turn 1
        if turn == 1 and not constraints.history:
            return "crane"

        # Step 2: Select evaluation pool (pruned by frequency if large for fast CPU inference)
        eval_cands = candidates
        if len(eval_cands) > 100:
            g_freq, p_freq = self.freq_solver.compute_frequencies(eval_cands)
            scored = [(w, self.freq_solver.score_word(w, g_freq, p_freq)) for w in eval_cands]
            scored.sort(key=lambda x: x[1], reverse=True)
            eval_cands = [w for w, _ in scored[:100]]

        # Step 3: Compute Neural Log Likelihood Prior
        history = list(constraints.history)
        state = WordleState(self.dictionary, history=history)
        prompt_seq = state.to_char_seq()
        
        neural_scores = self.neural_solver.model.score_candidates(
            prompt_seq, eval_cands, self.neural_solver.tokenizer, self.neural_solver.device
        )

        # Normalize neural scores to [0, 1] range
        min_n = min(neural_scores.values()) if neural_scores else 0.0
        max_n = max(neural_scores.values()) if neural_scores else 1.0
        norm_neural = {
            w: (s - min_n) / (max_n - min_n + 1e-8) for w, s in neural_scores.items()
        }

        # Step 4: Compute Information-Theoretic Entropy Scores
        entropy_scores = self.entropy_solver.get_all_entropy_scores(constraints, candidates=eval_cands)
        max_e = max(entropy_scores.values()) if entropy_scores else 1.0
        norm_entropy = {
            w: s / (max_e + 1e-8) for w, s in entropy_scores.items()
        }

        # Step 5: Compute Letter Frequency Scores
        g_freq, p_freq = self.freq_solver.compute_frequencies(eval_cands)
        raw_freq = {
            w: self.freq_solver.score_word(w, g_freq, p_freq) for w in eval_cands
        }
        max_f = max(raw_freq.values()) if raw_freq else 1.0
        norm_freq = {
            w: s / (max_f + 1e-8) for w, s in raw_freq.items()
        }

        # Step 6: Score Breakdown & Hybrid Weighted Ranking
        best_word = eval_cands[0]
        best_total_score = -float("inf")

        for word in eval_cands:
            n_score = norm_neural.get(word, 0.0)
            e_score = norm_entropy.get(word, 0.0)
            f_score = norm_freq.get(word, 0.0)

            total_score = (
                self.alpha * n_score
                + self.beta * e_score
                + self.gamma * f_score
            )

            if total_score > best_total_score:
                best_total_score = total_score
                best_word = word

        return best_word

    def get_candidate_rankings(
        self,
        constraints: SymbolicConstraints,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Returns top_k candidate scoring breakdowns for interpretability & visualization."""
        candidates = constraints.filter_candidates(self.dictionary.secret_words)
        if not candidates:
            candidates = constraints.filter_candidates(self.dictionary.all_valid_guesses)
        if not candidates:
            return []

        played_words = {guess.lower() for guess, _ in constraints.history}
        unplayed = [c for c in candidates if c not in played_words]
        if unplayed:
            candidates = unplayed

        # Fast path turn 1 or single candidate
        if len(candidates) == 1:
            return [{
                "word": candidates[0].upper(),
                "transformer_score": 1.0,
                "entropy_gain": 0.0,
                "freq_score": 1.0,
                "final_score": 1.0
            }]

        # Cap evaluation pool to 50 for instantaneous UI rendering
        eval_cands = candidates
        if len(eval_cands) > 50:
            g_freq, p_freq = self.freq_solver.compute_frequencies(eval_cands)
            scored = [(w, self.freq_solver.score_word(w, g_freq, p_freq)) for w in eval_cands]
            scored.sort(key=lambda x: x[1], reverse=True)
            eval_cands = [w for w, _ in scored[:50]]

        state = WordleState(self.dictionary, history=list(constraints.history))
        prompt_seq = state.to_char_seq()

        neural_scores = self.neural_solver.model.score_candidates(
            prompt_seq, eval_cands, self.neural_solver.tokenizer, self.neural_solver.device
        )
        min_n = min(neural_scores.values()) if neural_scores else 0.0
        max_n = max(neural_scores.values()) if neural_scores else 1.0
        norm_neural = {w: (s - min_n) / (max_n - min_n + 1e-8) for w, s in neural_scores.items()}

        entropy_scores = self.entropy_solver.get_all_entropy_scores(constraints, candidates=eval_cands)
        max_e = max(entropy_scores.values()) if entropy_scores else 1.0
        norm_entropy = {w: s / (max_e + 1e-8) for w, s in entropy_scores.items()}

        g_freq, p_freq = self.freq_solver.compute_frequencies(eval_cands)
        raw_freq = {w: self.freq_solver.score_word(w, g_freq, p_freq) for w in eval_cands}
        max_f = max(raw_freq.values()) if raw_freq else 1.0
        norm_freq = {w: s / (max_f + 1e-8) for w, s in raw_freq.items()}

        rankings = []
        for word in eval_cands:
            n_score = norm_neural.get(word, 0.0)
            e_score = norm_entropy.get(word, 0.0)
            f_score = norm_freq.get(word, 0.0)
            final_score = self.alpha * n_score + self.beta * e_score + self.gamma * f_score

            rankings.append({
                "word": word.upper(),
                "transformer_score": round(n_score, 3),
                "entropy_gain": round(entropy_scores.get(word, 0.0), 3),
                "freq_score": round(f_score, 3),
                "final_score": round(final_score, 3)
            })

        rankings.sort(key=lambda x: x["final_score"], reverse=True)
        return rankings[:top_k]
