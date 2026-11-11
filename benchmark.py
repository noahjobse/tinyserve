"""Benchmark: naive full-recompute generation vs the KV cache.

Both paths generate greedily (and are byte-identical, per the correctness test),
so this measures only the speed difference the cache buys.

    python benchmark.py
"""
import time

import torch

from model import GPT, GPTConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = GPTConfig()                      # the real ~51M config
model = GPT(cfg).to(device).eval()
prompt = torch.zeros((1, 8), dtype=torch.long, device=device)


@torch.no_grad()
def naive(idx, n):                     # recompute the whole sequence every token
    for _ in range(n):
        logits = model(idx[:, -cfg.block_size:])[0]
        idx = torch.cat((idx, logits[:, -1:, :].argmax(dim=-1)), dim=1)
    return idx


def timed(fn):
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return time.time() - t0


# warmup (first calls pay one-time setup cost)
naive(prompt.clone(), 5)
model.generate(prompt.clone(), 5)

print(f"device={device}, {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params, "
      f"prompt={prompt.size(1)} tokens")
print(f"{'new tokens':>11} | {'naive tok/s':>11} | {'cache tok/s':>11} | {'speedup':>7}")
for n in (64, 256):
    t_naive = timed(lambda: naive(prompt.clone(), n))
    t_cache = timed(lambda: model.generate(prompt.clone(), n))
    print(f"{n:>11} | {n / t_naive:>11.1f} | {n / t_cache:>11.1f} | {t_naive / t_cache:>6.1f}x")
