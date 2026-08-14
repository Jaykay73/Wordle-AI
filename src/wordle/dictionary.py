"""
Wordle Dictionary Management.
Loads, normalizes, and manages valid secret target solutions and allowed guess words.
Supports reproducible train/val/test splits of secret words to prevent data leakage.
"""

import os
import random
from typing import List, Set, Tuple


class WordleDictionary:
    def __init__(
        self,
        secrets_file: str = "data/raw/wordle_secret_words.txt",
        allowed_file: str = "data/raw/wordle_allowed_guesses.txt",
        seed: int = 42,
    ):
        self.secrets_file = secrets_file
        self.allowed_file = allowed_file
        self.seed = seed

        self.secret_words: List[str] = []
        self.allowed_words: List[str] = []
        self.all_valid_guesses: List[str] = []
        
        self._secret_set: Set[str] = set()
        self._valid_guess_set: Set[str] = set()

        self.train_secrets: List[str] = []
        self.val_secrets: List[str] = []
        self.test_secrets: List[str] = []

        self.load_words()
        self.create_splits()

    def _resolve_path(self, filepath: str) -> str:
        if os.path.exists(filepath):
            return filepath
        candidates = [
            os.path.join("..", filepath),
            os.path.join("/kaggle/working", filepath),
            os.path.join("/kaggle/working/wordle-ai", filepath),
            os.path.join(os.path.dirname(__file__), "..", "..", filepath),
        ]
        for cand in candidates:
            if os.path.exists(cand):
                return cand
        return filepath

    def _auto_download_wordlist(self, filepath: str):
        import urllib.request
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        if "secret" in filepath:
            url = "https://gist.githubusercontent.com/cfreshman/a03ef2cba789d8cf00c08f767e0fad7b/raw/wordle-answers-alphabetical.txt"
        else:
            url = "https://gist.githubusercontent.com/cfreshman/cdcdf777450c5b5301e439061d29694c/raw/wordle-allowed-guesses.txt"
        
        print(f"Downloading wordlist from {url} to {filepath}...", flush=True)
        urllib.request.urlretrieve(url, filepath)

    def _clean_word_list(self, filepath: str) -> List[str]:
        resolved = self._resolve_path(filepath)
        if not os.path.exists(resolved):
            try:
                self._auto_download_wordlist(filepath)
                resolved = filepath
            except Exception as e:
                raise FileNotFoundError(f"Could not find or download wordlist for '{filepath}': {e}")
        
        words = []
        with open(resolved, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip().lower()
                if len(word) == 5 and word.isalpha():
                    words.append(word)
        return sorted(list(set(words)))

    def load_words(self):
        self.secret_words = self._clean_word_list(self.secrets_file)
        allowed_extra = self._clean_word_list(self.allowed_file)

        self._secret_set = set(self.secret_words)
        # All valid guesses include secrets + allowed extra words
        self.all_valid_guesses = sorted(list(set(self.secret_words + allowed_extra)))
        self._valid_guess_set = set(self.all_valid_guesses)
        self.allowed_words = sorted(list(set(allowed_extra)))

    def create_splits(
        self,
        train_count: int = 1800,
        val_count: int = 250,
        test_count: int = 265
    ):
        """Creates reproducible, non-overlapping train/val/test splits of secret solution words."""
        rng = random.Random(self.seed)
        shuffled = list(self.secret_words)
        rng.shuffle(shuffled)

        total = len(shuffled)
        if train_count + val_count + test_count > total:
            # Adjust test count if needed
            test_count = total - train_count - val_count

        self.train_secrets = shuffled[:train_count]
        self.val_secrets = shuffled[train_count : train_count + val_count]
        self.test_secrets = shuffled[train_count + val_count : train_count + val_count + test_count]

    def is_valid_guess(self, word: str) -> bool:
        return word.lower() in self._valid_guess_set

    def is_secret_word(self, word: str) -> bool:
        return word.lower() in self._secret_set
