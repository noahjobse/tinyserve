# Building a GPT from scratch, and training it for a quarter

Last post I built up the idea the whole project rests on: why batching is the whole game. This one is the model itself. I built a small GPT by hand, trained it on my own laptop, and it learned to write coherent little stories. Total cost, start to finish, about 25 cents.

## What the model actually is

A GPT is simpler than it sounds once you take it apart. Mine is five kinds of pieces:

- **Embeddings:** turn each token into a vector, plus a position vector so the model knows word order.
- **Attention:** the step where tokens look at each other and mix in context. Each token asks "what am I looking for" (a query), every token advertises "here's what I am" (a key), and the relevant ones get blended in. This is the piece that does the real work.
- **The MLP:** after attention gathers context, each token processes what it gathered on its own.
- **A block:** attention plus MLP, with a couple of helpers to keep training stable. Mine stacks 8 of them.
- **The output head:** turns the final vectors into a score for every possible next token.

That's the whole architecture. 51 million parameters. It's the same machine as GPT-3.5, about 3,400 times smaller.

## The bug that announced itself

When I first ran training, the loss started at 84. It should start around 10.8. (That number is just the log of the vocabulary size, what pure random guessing scores. A fresh model knows nothing, so it should land exactly there.)

84 meant something was already wrong before training began. The cause: the output head shares its weights with the token embedding table, and the default setup for that table starts with values too large, so the model's first guesses came out wildly overconfident and wrong, which spikes the loss.

The fix was one line: start the weights small. After that it began at 10.84, right where it should, and trained cleaner. The lesson I keep relearning: watch the number that shouldn't be there.

## Training it

I trained on TinyStories, a dataset of simple children's stories built so even small models can produce real English. The loss fell steadily:

<!-- IMAGE: loss curve, 10.96 at iter 0 down to 2.03 at iter 5000, train and val lines tracking together. Produce in goals via chart_style.py on the identity palette (blue #1f6feb). -->
![Training loss curve: the loss falls from 10.96 at the start to 2.03 by 5000 iterations, with the training and validation lines tracking closely together the whole way.](loss-curve.png)

10.96 down to 2.03, with training and validation locked together the whole way, so nothing overfit. About 40 minutes of real GPU work on a 6GB laptop card.

## The time my screen died

Mid-training, my screen froze and went black. For a second I thought the machine had crashed and taken the run with it.

It hadn't. The process was still alive, GPU pinned at 99%. What actually happened: training had the VRAM pegged at 5.8 of 6.1 GB, and that same GPU also drives my display. When memory gets that tight under load, the Windows GPU driver does a reset, which blacks the screen for a moment but spares the compute job. It looked like a death. It was a hiccup.

It did expose a real gap, though. I was only saving a checkpoint at the very end, so if it actually had crashed, the whole run was gone. Now it saves every few hundred steps. That ties right back to the theme of the whole project: on a 6GB card, memory is the thing you are always fighting.

## What it wrote

Here's a story the finished model generated, cold:

> John was so excited. He couldn't wait to open it. He picked up his bed, put up a book and closed it. Suddenly, he heard a bright sound... Once upon a time, in a small garage. In the garage, there was a big brown box. The box had many colors and shapes. The kids were very excited...

Real grammar, character names, a narrative that mostly holds together, the TinyStories voice. It has rough edges (one invented word, a couple of non-sequiturs) that are exactly what a small model at a short training run looks like. But it's writing, from a thing I built out of nothing.

## What it cost

Electricity plus hardware wear, the whole run came to about **25 cents**.

For contrast: GPT-3.5's base model is estimated to have cost a few million dollars to train. The most expensive known training run, xAI's Grok 4, is estimated at around 490 million dollars and 310 gigawatt-hours of electricity, roughly what 30,000 homes use in a year. Mine used about a quarter of a kilowatt-hour, what a microwave burns in fifteen minutes.

Same architecture, though. Tokens in, attention, predict the next token, push the loss down. The billion-dollar run and my 25-cent run are the same machine at different scale.

## Next

So it works. It also recomputes the entire sequence from scratch for every single token it generates, which is painfully slow. The next post is where the real engineering starts: a KV cache, to stop throwing that work away, and the first speedup worth measuring.
