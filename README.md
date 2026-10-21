# tinyserve

A small LLM inference server, built from scratch. No serving frameworks, no vLLM, no TGI.

The plan, in order:

1. **Train** a tiny GPT (~45-50M params) on TinyStories, so there's a real model to serve.
2. **Inference engine** by hand: a KV cache and incremental decoding, replacing the naive recompute-everything generation loop.
3. **Serve** it: an HTTP server with request batching, a scheduler, token streaming, and a benchmark harness that measures throughput and tail latency (p50/p95/p99) under load.

Built in public, one piece at a time, measured.

## Status

Early. Scaffolding up; model next.
