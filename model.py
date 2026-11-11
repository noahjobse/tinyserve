# Dataclass auto gens boiler plate for classes
from dataclasses import dataclass

# Torch is an ML library
import torch
# Neural-net builfing blocks called 'nn'
import torch.nn as nn
import torch.nn.functional as F
import math


# GPT config settings using dataclass
@dataclass
class GPTConfig:
    # How many tokens the model knows
    # Set by GPT-2s BPE vocab of 50257 tokens
    vocab_size: int = 50257

    # The longest sequence of tokens it can handle
    # Choosen for data and memory - in this case tinystories
    block_size: int = 512 

    # ALL: Capacity dials chosen based on hardware
    n_layer: int = 8
    # Attention heads per block
    n_head: int = 8
    # Width of each tokens vector
    # embed width MUST divide cleanly by attention heads
    n_embd: int = 512
   
    # A training knob: randomly drops values to prevent overfitting - set to off for now
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.qkv  = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))

    def forward(self, x, cache=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)                        # each (B, T, C)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        if cache is not None:                        # decode: prepend the cached past
            past_k, past_v = cache
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        Tk = k.size(2)                               # total keys = past + new
        scores = (q @ k.transpose(-2, -1)) * self.head_size ** -0.5   # (B, nh, T, Tk)
        scores = scores.masked_fill(self.tril[Tk - T:Tk, :Tk] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = (weights @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out), (k, v)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2  = nn.LayerNorm(config.n_embd)
        self.mlp  = MLP(config)

    def forward(self, x, cache=None):
        a, new_cache = self.attn(self.ln1(x), cache)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, new_cache


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, caches=None):
        B, T = idx.shape
        past_len = caches[0][0].size(2) if caches is not None else 0
        tok_emb = self.wte(idx)
        pos = torch.arange(past_len, past_len + T, device=idx.device)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb
        new_caches = []
        for i, block in enumerate(self.blocks):
            x, nc = block(x, caches[i] if caches is not None else None)
            new_caches.append(nc)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, new_caches

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        logits, caches = self(idx)                     # prefill: run the whole prompt once
        for _ in range(max_new_tokens):
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)   # greedy
            idx = torch.cat((idx, next_id), dim=1)
            logits, caches = self(next_id, caches)     # decode: one token, reuse the cache
        return idx
