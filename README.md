# tinyserve

A small LLM inference server, built from scratch and measured the way an engineer would measure it. No serving frameworks, no vLLM, no TGI.

Two halves, both built by hand: a tiny GPT, and the server that runs it fast under load. The interesting work is where they meet, so the whole thing is treated as a performance-engineering problem from the start.

The one idea that drives every decision: during generation, decode is memory-bandwidth-bound and prefill is compute-bound. Batching requests together is the whole game for throughput, and this project's job is to prove that with real numbers.

## The plan, in layers

0. **Roofline and baselines.** Before building, compute what the hardware (an RTX 4050, 6GB) can actually do, and stand up reference baselines (HuggingFace naive `generate`, and vLLM) on the same model and hardware. Set the targets we are chasing before writing the engine.
1. **Train a model.** A tiny GPT (~45-50M params) on TinyStories, by hand, so there is a real model to serve.
2. **Inference engine.** A block-based KV cache (PagedAttention-style) and continuous batching (Orca-style), replacing the naive recompute-everything loop. Prefill/decode split, incremental decoding.
3. **Serve it.** An HTTP server, a scheduler, request batching, token streaming, and a load harness that drives realistic (Poisson) traffic and reports the full distribution: throughput, TTFT, TPOT (time per output token), p50/p95/p99, and goodput under an SLO.

## How it is validated (no self-grading)

Three independent judges:
- **Correctness oracle:** greedy decode through the fast path (KV cache + batching) must produce byte-identical output to the naive full-recompute path.
- **Roofline:** measured throughput is checked against the predicted ceiling, as proof the machine is understood.
- **Baselines:** curves are compared to HF-naive and vLLM on the same hardware. Beating vLLM is out of scope; the sanity check is landing in the known range.

## Useful to others

A reproducible benchmark harness anyone can clone and run on their own GPU, plus a writeup of each layer that teaches the core insight (why decode is memory-bound, and why batching wins), mistakes included.

## Status

Early. Data prep done, model next. Built in public, one piece at a time, measured.
