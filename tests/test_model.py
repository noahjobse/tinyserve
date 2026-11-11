"""Tiny shape checks for the model as we build it. Run from the repo root:

    .venv/bin/python tests/test_model.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model import GPT, GPTConfig, CausalSelfAttention, MLP, Block


def test_gpt_logits_shape():
    cfg = GPTConfig(vocab_size=100, block_size=16, n_layer=2, n_head=4, n_embd=32)
    model = GPT(cfg)
    idx = torch.randint(0, 100, (2, 5))  # batch of 2 sequences, 5 tokens each
    logits, _ = model(idx)
    assert logits.shape == (2, 5, 100), f"expected (2, 5, 100), got {tuple(logits.shape)}"
    print("ok: gpt logits shape is", tuple(logits.shape))


def test_attention_shape():
    cfg = GPTConfig(vocab_size=100, block_size=16, n_embd=32, n_head=4)
    attn = CausalSelfAttention(cfg)
    x = torch.randn(2, 5, 32)  # batch 2, seq 5, each token a 32-wide vector
    out, _ = attn(x)
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
    out, _ = block(x)
    assert out.shape == (2, 5, 32), f"expected (2, 5, 32), got {tuple(out.shape)}"
    print("ok: block output shape is", tuple(out.shape))


def test_kv_cache_matches_naive():
    # the correctness oracle: cached generation must be byte-identical to naive recompute
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=50, block_size=32, n_layer=2, n_head=2, n_embd=16)
    model = GPT(cfg).eval()
    idx = torch.randint(0, 50, (1, 4))
    cached = model.generate(idx.clone(), max_new_tokens=12)
    naive = idx.clone()
    for _ in range(12):
        logits, _ = model(naive)
        naive = torch.cat((naive, logits[:, -1, :].argmax(dim=-1, keepdim=True)), dim=1)
    assert torch.equal(cached, naive), "KV-cache output diverged from naive recompute"
    print("ok: kv-cache generation is byte-identical to naive")


if __name__ == "__main__":
    test_attention_shape()
    test_mlp_shape()
    test_block_shape()
    test_gpt_logits_shape()
    test_kv_cache_matches_naive()
