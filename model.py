"""tinyserve's GPT model, built one piece at a time.

Right now this has only the input stage: turning token IDs into vectors
(token embedding + position embedding). The transformer blocks, final
LayerNorm, and output head come in later pieces.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GPTConfig:
    vocab_size: int = 50257   # GPT-2 BPE vocab size
    block_size: int = 512     # max context length (how many tokens it can see at once)
    n_layer: int = 8          # number of transformer blocks (added later)
    n_head: int = 8           # attention heads per block (added later)
    n_embd: int = 512         # width of each token vector
    dropout: float = 0.0


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        # Two lookup tables, both learned during training:
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)  # token id -> vector
        self.wpe = nn.Embedding(config.block_size, config.n_embd)  # position  -> vector
        # (coming later: self.blocks, self.ln_f, self.lm_head)

    def forward(self, idx):
        # idx: a (B, T) tensor of token IDs. B = batch size, T = sequence length.
        B, T = idx.shape
        tok_emb = self.wte(idx)                   # (B, T, n_embd): each token id -> its vector
        pos = torch.arange(T, device=idx.device)  # (T,): positions 0, 1, ..., T-1
        pos_emb = self.wpe(pos)                   # (T, n_embd): each position -> its vector
        x = tok_emb + pos_emb                     # (B, T, n_embd): add, pos broadcasts over the batch
        return x
        