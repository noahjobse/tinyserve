"""Training loop for tinyserve's GPT.

Defaults are the REAL run: the full ~50M model on TinyStories, bf16 on the GPU, with the
standard recipe (LR warmup + cosine decay, gradient clipping, resumable checkpoints).
Every knob is env-overridable; a quick CPU smoke test is e.g.:
    DATASET=shakespeare BLOCK_SIZE=64 N_LAYER=2 N_EMBD=64 MAX_ITERS=20 python train.py

Real run (on the GPU):   python train.py
Resume a crashed run:    RESUME=1 python train.py
"""
import math
import os
from contextlib import nullcontext
from dataclasses import asdict

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig

def _int(k, d): return int(os.environ.get(k, d))

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CKPT = os.environ.get("CKPT", os.path.join(DATA_DIR, "..", "ckpt.pt"))
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
grad_clip = 1.0
warmup_iters = _int("WARMUP_ITERS", 100)
min_lr = lr / 10

device = "cuda" if torch.cuda.is_available() else "cpu"
ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
       if device == "cuda" else nullcontext())

def load(split):
    return np.memmap(os.path.join(DATA_DIR, f"{dataset}_{split}.bin"), dtype=np.uint16, mode="r")

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

def get_lr(it):
    if it < warmup_iters:
        return lr * (it + 1) / (warmup_iters + 1)
    if it >= max_iters:
        return min_lr
    ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    return min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (lr - min_lr)

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
        with ctx:
            logits = model(idx[:, -block_size:])
        probs = F.softmax(logits[:, -1, :], dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
    model.train()
    return enc.decode(idx[0].tolist())

config = GPTConfig(block_size=block_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd)
model = GPT(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

start_iter = 0
if os.environ.get("RESUME") and os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location=device, weights_only=True)
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    start_iter = ck["iter"]
    print(f"resumed from iter {start_iter}")

def save(it):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": asdict(config), "iter": it}, CKPT)

print(f"dataset={dataset}, device={device}, params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
for it in range(start_iter, max_iters + 1):
    for g in optimizer.param_groups:
        g["lr"] = get_lr(it)
    if it % eval_interval == 0:
        losses = estimate_loss()
        print(f"iter {it:5d} | lr {get_lr(it):.2e} | train {losses['train']:.3f} | val {losses['val']:.3f}")
        save(it)
    x, y = get_batch("train")
    with ctx:
        loss = loss_fn(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

save(max_iters)
print("\nsaved ckpt\n\n--- sample ---")
print(sample())
