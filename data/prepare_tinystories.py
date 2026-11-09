"""Data prep for the real run: TinyStories.

Downloads the TinyStories dataset, tokenizes it with the GPT-2 BPE vocab, and
writes two flat uint16 token files:
    data/tinystories_train.bin
    data/tinystories_val.bin

Stories are separated by the GPT-2 end-of-text token (50256) so the model learns
where one story ends and the next begins. Writes incrementally, so memory stays
flat even though the training split is hundreds of millions of tokens.

Run once (downloads ~1 GB, takes a few minutes):
    python data/prepare_tinystories.py
"""
import os

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token  # 50256, the story boundary marker

FLUSH_EVERY = 1_000_000  # tokens buffered before writing to disk


def tokenize_split(dataset_split, out_path):
    total = 0
    buf = []
    with open(out_path, "wb") as f:
        for example in tqdm(dataset_split):
            buf.extend(enc.encode_ordinary(example["text"]))
            buf.append(EOT)
            if len(buf) >= FLUSH_EVERY:
                np.array(buf, dtype=np.uint16).tofile(f)
                total += len(buf)
                buf = []
        if buf:
            np.array(buf, dtype=np.uint16).tofile(f)
            total += len(buf)
    print(f"wrote {out_path}: {total:,} tokens")


if __name__ == "__main__":
    ds = load_dataset("roneneldan/TinyStories")
    tokenize_split(ds["train"], os.path.join(HERE, "tinystories_train.bin"))
    tokenize_split(ds["validation"], os.path.join(HERE, "tinystories_val.bin"))
