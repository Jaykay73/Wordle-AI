"""
Unit tests for Synthetic Trajectory Dataset Generation & Data Leakage Prevention.
"""

from src.wordle.dictionary import WordleDictionary
from src.training.generation import TrajectoryGenerator
from src.training.dataset import CharTokenizer, WordleTrajectoryDataset


def test_trajectory_generator_leakage_prevention():
    dict_mgr = WordleDictionary(seed=42)
    generator = TrajectoryGenerator(dict_mgr, seed=42)

    train_samples = generator.generate_dataset(dict_mgr.train_secrets, num_samples_target=20)
    assert len(train_samples) == 20

    train_secret_set = set(dict_mgr.train_secrets)
    test_secret_set = set(dict_mgr.test_secrets)

    # 1. Assert zero secret set overlap
    assert len(train_secret_set.intersection(test_secret_set)) == 0

    # 2. Assert secret word NEVER appears in prompt text before <NEXT>
    for s in train_samples:
        assert s.secret_word not in s.prompt_seq.split("<NEXT>")[0]
        # Assert test secret is never used as target in training set
        assert s.secret_word not in test_secret_set


def test_char_tokenizer():
    tokenizer = CharTokenizer()
    prompt = "<GUESS>crane</GUESS><FEEDBACK>01200</FEEDBACK><NEXT>"
    target = "slate"

    token_ids = tokenizer.encode_sequence(prompt, target)
    assert len(token_ids) > 0
    assert token_ids[0] == tokenizer.bos_id
    assert token_ids[-1] == tokenizer.eos_id

    decoded = tokenizer.decode_ids(token_ids)
    assert "crane" in decoded
    assert "slate" in decoded
