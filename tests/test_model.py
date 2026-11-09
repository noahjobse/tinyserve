"""Tiny shape checks for the model as we build it. Run from the repo root:

    .venv/bin/python tests/test_model.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model import GPT, GPTConfig, MultiHeadAttention, MLP, Block


def test_gpt_logits_shape():
    cfg = GPTConfig(vocab_size=100, block_size=16, n_layer=2, n_head=4, n_embd=32)
    model = GPT(cfg)
    idx = torch.randint(0, 100, (2, 5))  # batch of 2 sequences, 5 tokens each
    logits = model(idx)
    assert logits.shape == (2, 5, 100), f"expected (2, 5, 100), got {tuple(logits.shape)}"
    print("ok: gpt logits shape is", tuple(logits.shape))


def test_attention_shape():
    cfg = GPTConfig(vocab_size=100, block_size=16, n_embd=32, n_head=4)
    mha = MultiHeadAttention(cfg)
    x = torch.randn(2, 5, 32)  # batch 2, seq 5, each token a 32-wide vector
    out = mha(x)
    assert out.shape == (2, 5, 32), f"expected (2, 5, 32), got {tuple(out.shape)}"
    print("ok: attention output shape is", tuple(out.shape))


def test_mlp_shape():
    cfg = GPTConfig(n_embd=32)
    mlp = MLP(cfg)
    x = torch.randn(2, 5, 32)
    out = mlp(x)
    assert out.shape == (2, 5, 32), f"expected (2, 5, 32), got {tuple(out.shape)}"
    print("ok: mlp output shape is", tuple(out.shape))


def test_block_shape():
    cfg = GPTConfig(block_size=16, n_embd=32, n_head=4)
    block = Block(cfg)
    x = torch.randn(2, 5, 32)
    out = block(x)
    assert out.shape == (2, 5, 32), f"expected (2, 5, 32), got {tuple(out.shape)}"
    print("ok: block output shape is", tuple(out.shape))


if __name__ == "__main__":
    test_attention_shape()
    test_mlp_shape()
    test_block_shape()
    test_gpt_logits_shape()
