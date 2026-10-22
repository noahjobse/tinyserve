# Building an LLM inference server from scratch

## Why I'm doing this

I just finished building a RAG agent from scratch, no frameworks, every piece by hand. This is the start of the next one, and it goes after two things at once.

I'm building a small LLM inference server from scratch. Two halves: the model, trained by hand, and the server that runs it fast under load. No vLLM, no TGI, no serving framework. One half is the AI, how a language model actually works on the inside. The other half is the platform, the systems engineering that turns "it generates text" into "it serves a crowd of requests without falling over." I want both, because the interesting work lives exactly where they meet.

Same reason as last time. I've leaned on this infrastructure for years and couldn't have told you how a single part of it works. What attention is really computing. What a KV cache actually caches. Why batching requests together is the whole game for throughput. Where tail latency comes from when a server is slammed. I could say those words. I couldn't build any of it. Same fix as the RAG project: build it by hand, measure it honestly, and don't ship anything I can't explain.

## What it actually is, in three layers

Three pieces, and I'm building all of them. The first is the AI side, the last is the platform side, and the middle one is the bridge between them.

**1. Train a model (the AI side).** Before you can serve a model you need one. I'll train a tiny GPT, around 45 to 50 million parameters, on TinyStories, a dataset built so small models still produce real English. Attention, the MLP, tokenization, the training loop, all by hand. A model that's actually mine, trained on a 6GB laptop GPU.

**2. The inference engine (the bridge).** This is where the two sides meet. The naive way to generate text recomputes the entire sequence for every single token, which is O(n^2) work. A real server avoids it with a KV cache: the attention state from the tokens so far, held and reused instead of recomputed. That's an ML structure and a performance trick at the same time. I'll write the cache, the prefill-then-decode split, and incremental decoding by hand. This is what turns "it generates text" into "it generates text fast."

**3. The server (the platform side).** Then I make it serve. An HTTP API, many requests at once, batching them so the GPU stays busy, continuous batching so finished requests get swapped out mid-flight and the GPU never idles, a scheduler deciding what runs when, KV-cache memory managed inside a tight 6GB budget, token streaming, and a benchmark harness measuring the things that actually matter: throughput, time to first token, and tail latency (p50, p95, p99) under load.

## What I'm borrowing, and what I'm not

I'm using Karpathy's nanoGPT as the reference for the model and training. About 750 readable lines, it covers the AI side well, plus the naive version of the engine.

What it leaves out is the entire reason I'm doing this. nanoGPT has no KV cache, no server, no batching, no scheduler, no streaming, and no serving benchmarks. Its generation recomputes everything, every token. So the model and training are a guided build, and the engine and the serving layer, the platform side I actually care about, are mine to write from nothing. Same rule as the RAG project: lean on a library only for the stuff that isn't the point, and write everything that teaches me something.

## How it'll go

Same as last time. In public, one piece at a time, mistakes included. I'll write up each layer as I build it.

Rough plan:

1. Train the model (and what 45M parameters can actually say)
2. The inference engine: KV cache, prefill and decode
3. Serving: HTTP, batching, streaming
4. Continuous batching and the scheduler
5. Benchmarks: throughput and tail latency under load

Bottom-up on purpose. You can't measure a server you haven't built, and you can't build one without a model to run.

And there's a payoff I'm building toward: a live demo on my site where you can fire a burst of traffic at the running server and watch the throughput, first-token time, and tail latency move in real time. The whole point of the platform side, made something you can poke at.

If you want to watch a model, and the infrastructure to serve it, get built from the ground up and measured at every step, follow along. First post is the model.
