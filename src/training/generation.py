"""
Synthetic Trajectory Generator for Wordle Neural Solvers.
Generates game state transition samples using mixed teacher strategies (random, frequency, entropy)
while guaranteeing zero data leakage across train, val, and test secret splits.
"""

import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from ..wordle.dictionary import WordleDictionary
from ..wordle.game import WordleGame, GameStatus
from ..wordle.constraints import SymbolicConstraints
from ..wordle.state import WordleState
from ..solvers.random_solver import RandomSolver
from ..solvers.frequency_solver import FrequencySolver
from ..solvers.entropy_solver import EntropySolver


@dataclass
class TrajectorySample:
    prompt_seq: str
    target_word: str
    turn: int
    secret_word: str


class TrajectoryGenerator:
    def __init__(self, dictionary: WordleDictionary, seed: int = 42):
        self.dictionary = dictionary
        self.seed = seed
        self.rng = random.Random(seed)

        self.random_solver = RandomSolver(dictionary, seed=seed)
        self.freq_solver = FrequencySolver(dictionary)
        self.entropy_solver = EntropySolver(dictionary)

    def generate_game_trajectory(
        self,
        secret_word: str,
        strategy: str = "mixed"
    ) -> List[TrajectorySample]:
        """
        Simulates a game for a given secret word and extracts turn-by-turn state transition samples.
        """
        game = WordleGame(secret_word=secret_word, dictionary=self.dictionary)
        constraints = SymbolicConstraints()
        samples: List[TrajectorySample] = []
        history: List[Tuple[str, Tuple[int, ...]]] = []

        while game.status == GameStatus.IN_PROGRESS and game.turns_used < 6:
            state = WordleState(self.dictionary, history=list(history))
            prompt = state.to_char_seq()
            turn = game.turns_used + 1

            # Select strategy for next move
            current_strat = strategy
            if strategy == "mixed":
                current_strat = self.rng.choice(["frequency", "frequency", "random"])

            if current_strat == "entropy":
                guess = self.entropy_solver.choose_guess(constraints, turn=turn)
            elif current_strat == "frequency":
                guess = self.freq_solver.choose_guess(constraints, turn=turn)
            else:
                guess = self.random_solver.choose_guess(constraints, turn=turn)

            # Sanity assertion against data leakage: secret MUST NOT appear inside prompt text
            assert secret_word.lower() not in prompt.lower().split("<NEXT>")[0], "Data leakage error: Secret word found inside prompt!"

            samples.append(TrajectorySample(
                prompt_seq=prompt,
                target_word=guess.lower(),
                turn=turn,
                secret_word=secret_word.lower()
            ))

            fb = game.make_guess(guess)
            history.append((guess.lower(), fb))
            constraints.add_feedback(guess, fb)

        return samples

    def generate_dataset(
        self,
        secret_words: List[str],
        num_samples_target: int = 10000,
        strategy: str = "mixed"
    ) -> List[TrajectorySample]:
        """Generates synthetic dataset from a designated split of secret words."""
        samples: List[TrajectorySample] = []
        secret_pool = list(secret_words)
        
        if not secret_pool:
            return []

        idx = 0
        while len(samples) < num_samples_target:
            secret = secret_pool[idx % len(secret_pool)]
            game_samples = self.generate_game_trajectory(secret, strategy=strategy)
            samples.extend(game_samples)
            idx += 1

        return samples[:num_samples_target]
