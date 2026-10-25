"""Tiny shape checks for the model as we build it. Run from the repo root:

    .venv/bin/python tests/test_model.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model import GPT, GPTConfig


def test_embedding_shape():
    cfg = GPTConfig(vocab_size=100, block_size=16, n_embd=8)
    model = GPT(cfg)
    idx = torch.randint(0, 100, (2, 5))  # batch of 2 sequences, 5 tokens each
    x = model(idx)
    assert x.shape == (2, 5, 8), f"expected (2, 5, 8), got {tuple(x.shape)}"
    print("ok: embedding output shape is", tuple(x.shape))


if __name__ == "__main__":
    test_embedding_shape()
