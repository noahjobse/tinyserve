# Building an LLM inference server from scratch

This kicks off my next build-in-public project, and I want to open it by teaching the first thing I learned, because it turned out to be the idea the whole project rests on.

Short version of what I'm doing: building a small LLM inference server from scratch. No vLLM, no TGI, no serving framework. Two halves, both by hand: a tiny language model, and the server that runs it fast under load. My last project was a RAG agent built the same way, no frameworks, every piece by hand. This one aims higher.

## Why

I've leaned on inference infrastructure for years and couldn't have told you how a single part of it worked. What a KV cache actually caches. Why batching requests is the whole game for throughput. Where tail latency comes from when a server is under load. I could say those words. I couldn't build any of it. Same fix as the RAG project: build it by hand, measure it honestly, and don't ship anything I can't explain.

## The two halves

What makes this interesting is that it sits right where two kinds of engineering meet.

The ML side: I train a tiny GPT from scratch, attention, the MLP, the training loop, all by hand, so there's a real model that's mine to serve.

The platform side: I wrap it in a real server. Concurrency, request batching, a scheduler, token streaming, memory management on a tight 6GB budget, and a benchmark harness that measures what actually matters under load (throughput, time to first token, tail latency).

The model is the AI. The server is the systems engineering. The hard part is in the middle, where they meet.

## The first thing I learned: why batching is the whole game

This is the first big ML concept that actually clicked for me, and it set the target for everything else.

When a model generates text, it makes one token at a time. To produce each token, the GPU loads the entire model's weights out of its memory and runs the math on them.

Here's the part I didn't expect: the math is almost nothing. For a single token, the GPU does a tiny amount of arithmetic on each weight and finishes almost instantly. So it does the math in a flash and then waits for the next load of weights to arrive from memory.

That's what "memory-bound" means. The bottleneck is how fast the GPU can read the weights out of its own memory, and the expensive compute units sit idle most of the time, waiting. It helped me to split the GPU into two things: compute (how fast it does math) and memory bandwidth (how fast it reads its own memory). During decode, bandwidth is pegged and compute is idle.

Which is why batching is basically free. The weights get loaded no matter what, whether you're serving one request or many. So if you batch a bunch of requests together and run them through the same loaded weights at once, you do far more useful work for the same memory cost. One request: load all the weights, do one token's worth of math, done, almost all of that load wasted. Batch 47 requests: load the weights once, do 47 tokens' worth of math on them. Same memory traffic, 47 times the work. The compute units that were idling now have something to chew on.

There's a clean way to see where the free lunch ends, called the roofline. Picture a graph with two ceilings: a sloped line for memory bandwidth, a flat line for peak compute. Where they meet is the ridge point, measured in FLOPs per byte (math done per byte read). Below it you're memory-bound and batching is nearly free. Above it you're compute-bound and batching stops helping.

I ran the numbers for my laptop GPU, an RTX 4050:
- Memory bandwidth: 192 GB/s
- FP16 compute: 8.986 TFLOP/s
- Ridge point: 8.986e12 / 192e9 ≈ 47 FLOPs per byte

![Roofline plot for the RTX 4050 Laptop GPU: a sloped memory-bandwidth line (192 GB/s) rises until it meets the flat compute ceiling (8.986 FP16 TFLOP/s) at the ridge point near 47 FLOPs per byte. A single request sits far down the slope in the memory-bound region at about 1 FLOP per byte; batching raises arithmetic intensity up the slope to the ridge, where the workload turns compute-bound.](part1-roofline.png)

Since the work per byte roughly equals the batch size here, that ridge lands at a batch of about 47. So on my 4050 I can batch up to around 47 requests almost for free. Below that I'm memory-bound with spare compute going to waste; past it, each extra request costs real compute time. 47 is the most free throughput I get before the crossover, and it's the number the whole server is organized around. The scheduler and the batching logic exist to keep the batch as full as possible, up toward that ridge.

I couldn't have told you any of this a week ago. One honest caveat I hit at the end: this first-order number ignores the KV cache, which also eats memory bandwidth and grows with batch size and context length. That pushes the real ridge around, and it's a problem for a later post.

## The roadmap

Built bottom-up, one layer at a time, in public:

0. **Roofline and baselines.** Figure out what the hardware can actually do (above) and set the targets before building anything.
1. **Train the model.** A tiny GPT (~45-50M params) on TinyStories, so there's a real model to serve.
2. **The inference engine.** A KV cache and continuous batching, replacing the naive recompute-everything loop. This is the core.
3. **The server.** An HTTP API, a scheduler, streaming, and a load harness that reports the full latency distribution (p50/p95/p99, time to first token, goodput under load).

The payoff I'm building toward: a live demo on my site where you can throw traffic at the running server and watch the throughput and latency move in real time.

## Timeline

Roughly one layer every couple of weeks, with a writeup for each as I go, aiming to have the whole thing serving and benchmarked in a couple of months. Mistakes included, because that's where the learning is.

## Follow along

If you want to watch a model, and the infrastructure to serve it, get built from the ground up and measured at every step, follow along. Next post: training the model.
