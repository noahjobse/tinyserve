"""Training loop for tinyserve's GPT.

Defaults are the REAL run: the full ~50M model on TinyStories, bf16 on the GPU.
Every knob is overridable by env var, so a quick CPU smoke test is e.g.:
    DATASET=shakespeare BLOCK_SIZE=64 N_LAYER=2 N_EMBD=64 MAX_ITERS=20 python train.py

Real run (on the GPU):
    python train.py
"""
import os
from contextlib import nullcontext

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig

# --- config (env-overridable) ---
def _int(k, d): return int(os.environ.get(k, d))

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
dataset = os.environ.get("DATASET", "tinystories")
block_size = _int("BLOCK_SIZE", 512)
batch_size = _int("BATCH_SIZE", 16)
n_layer = _int("N_LAYER", 8)
n_head = _int("N_HEAD", 8)
n_embd = _int("N_EMBD", 512)
max_iters = _int("MAX_ITERS", 5000)
eval_interval = _int("EVAL_INTERVAL", 500)
eval_iters = _int("EVAL_ITERS", 100)
lr = float(os.environ.get("LR", 3e-4))

device = "cuda" if torch.cuda.is_available() else "cpu"
ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
       if device == "cuda" else nullcontext())

# --- data ---
def load(split):
    path = os.path.join(DATA_DIR, f"{dataset}_{split}.bin")
    return np.memmap(path, dtype=np.uint16, mode="r")

train_data, val_data = load("train"), load("val")

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

def loss_fn(logits, targets):
    B, T, V = logits.shape
    return F.cross_entropy(logits.view(B * T, V), targets.view(B * T))

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            with ctx:
                losses[k] = loss_fn(model(x), y).item()
        out[split] = losses.mean().item()
    model.train()
    return out

@torch.no_grad()
def sample(max_new_tokens=200):
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    idx = torch.tensor([[enc.eot_token]], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        with ctx:
            logits = model(idx_cond)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
    model.train()
    return enc.decode(idx[0].tolist())

# --- model + optimizer ---
config = GPTConfig(block_size=block_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd)
model = GPT(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

print(f"dataset={dataset}, device={device}, params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
for it in range(max_iters + 1):
    if it % eval_interval == 0:
        losses = estimate_loss()
        print(f"iter {it:5d} | train {losses['train']:.3f} | val {losses['val']:.3f}")
    x, y = get_batch("train")
    with ctx:
        loss = loss_fn(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save({"model": model.state_dict(), "config": config}, os.path.join(DATA_DIR, "..", "ckpt.pt"))
print("\nsaved ckpt.pt\n\n--- sample ---")
print(sample())
