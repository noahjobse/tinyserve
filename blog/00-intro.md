# Building an LLM inference server from scratch

I just finished a five-part series building a RAG agent from scratch, no frameworks, every piece by hand. This is the start of the next one, and it aims higher.

I'm building a small LLM inference server from scratch: the thing that loads a language model, takes requests, and serves generated text under load. No vLLM, no TGI, no serving framework. I want to understand the machine that runs underneath every AI product I've used, by building it myself and measuring it.

## Why do this

I've leaned on inference infrastructure for years and couldn't have told you how a single part of it works. What a KV cache actually caches. Why batching requests together is the whole game for throughput. Where tail latency comes from when a server is under load. I could name these things. I could not build them.

That's the same gap that drove the RAG project, and the same fix: build it by hand, measure it honestly, and don't ship anything I can't explain.

## What it is, in three layers

The project splits into three pieces, and I'm building all of them.

**1. Train a model.** Before you can serve a model, you need one. I'll train a tiny GPT (around 45 to 50 million parameters) on TinyStories, a dataset built so small models still produce coherent English. That gives me a real model that's mine, trained on a 6GB laptop GPU.

**2. The inference engine.** This is where it gets interesting. The naive way to generate text recomputes the entire sequence for every single token, which is O(n^2) work. A real server avoids that by keeping a KV cache: the attention state from previous tokens, held and reused rather than recomputed. I'll write the cache, the prefill-then-decode split, and incremental decoding by hand. This is the part that turns "it generates text" into "it generates text fast."

**3. The server.** Then I make it serve. An HTTP API, multiple concurrent requests, batching them together so the GPU stays busy, a scheduler deciding what runs when, token streaming so output appears as it is generated, and a benchmark harness that measures the things that actually matter: throughput, time to first token, and tail latency (p50, p95, p99) under load.

## What I'm borrowing, and what I'm not

I'm using Karpathy's nanoGPT as a reference for the model and training. It's about 750 readable lines, and it covers layer 1 completely and the naive version of layer 2.

What it leaves out is the entire reason I'm doing this. nanoGPT has no KV cache, no server, no batching, no scheduler, no streaming, and no serving benchmarks. Its text generation recomputes everything every token. So the model and training are a guided reimplementation, and the inference engine and the serving layer, the parts I actually care about, are mine to build from nothing.

## How it will go

Same as last time. In public, one piece at a time, mistakes included. I'll write up each layer as I build it: the training run and what a tiny model can and cannot do, the KV cache and the speedup it buys, and the serving work where the real engineering lives.

If you want to watch AI infrastructure get built from the ground up and measured at each step, follow along.
