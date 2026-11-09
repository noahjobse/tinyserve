"""Training loop for tinyserve's GPT.

Defaults here are a small, fast SMOKE-TEST config (proves the model learns on
tiny-shakespeare on a CPU in a couple of minutes). For the real TinyStories run
on the GPU, bump block_size / n_layer / n_head / n_embd / max_iters back up.

    python train.py
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig

# --- config (smoke-test sized) ---
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
device = "cuda" if torch.cuda.is_available() else "cpu"
block_size = 64
batch_size = 16
n_layer, n_head, n_embd = 4, 4, 128
max_iters = 500
eval_interval = 100
eval_iters = 20
lr = 3e-4

# --- data: memory-map the token streams, grab random chunks ---
def load(split):
    path = os.path.join(DATA_DIR, f"shakespeare_{split}.bin")
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
            losses[k] = loss_fn(model(x), y).item()
        out[split] = losses.mean().item()
    model.train()
    return out

# --- model + optimizer ---
config = GPTConfig(block_size=block_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd)
model = GPT(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

print(f"device={device}, params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
for it in range(max_iters + 1):
    if it % eval_interval == 0:
        losses = estimate_loss()
        print(f"iter {it:4d} | train {losses['train']:.3f} | val {losses['val']:.3f}")
    x, y = get_batch("train")
    loss = loss_fn(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print("done")
