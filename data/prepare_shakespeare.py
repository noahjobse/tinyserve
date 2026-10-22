"""Data prep for the tiny-shakespeare smoke test.

Turns a plain-text file into the two things training needs on disk:
    data/shakespeare_train.bin   (~90% of the tokens)
    data/shakespeare_val.bin     (~10%, held out to measure overfitting)

Each .bin is just a flat array of token IDs (uint16) with no structure: the
training loop reads random chunks straight out of it. Run once:

    python data/prepare_shakespeare.py
"""
import os
import urllib.request

import numpy as np
import tiktoken

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(HERE, "shakespeare_input.txt")
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# 1. Download the raw text once (~1 MB: all of Shakespeare's plays as one file).
if not os.path.exists(INPUT_PATH):
    print("downloading tiny-shakespeare ...")
    urllib.request.urlretrieve(DATA_URL, INPUT_PATH)

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    text = f.read()
print(f"loaded {len(text):,} characters")

# 2. Split BEFORE tokenizing, by character position, so val is text the model
#    never trained on. 90/10 is the usual split for a dataset this small.
n = len(text)
train_text = text[: int(n * 0.9)]
val_text = text[int(n * 0.9) :]

# 3. Tokenize with the GPT-2 BPE vocab (~50k pieces). encode_ordinary skips
#    the special end-of-text token; we add our own structure later.
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_text)
val_ids = enc.encode_ordinary(val_text)
print(f"train: {len(train_ids):,} tokens   val: {len(val_ids):,} tokens")

# 4. Save as uint16. The GPT-2 vocab maxes at 50256, which fits in uint16
#    (0..65535), so each token is 2 bytes instead of 8. Half the disk, fast load.
np.array(train_ids, dtype=np.uint16).tofile(os.path.join(HERE, "shakespeare_train.bin"))
np.array(val_ids, dtype=np.uint16).tofile(os.path.join(HERE, "shakespeare_val.bin"))
print("wrote shakespeare_train.bin and shakespeare_val.bin")
