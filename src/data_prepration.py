# data_preparation.py

"""Data preparation for transformer model."""

import torch
from urllib.request import urlopen

# Downloading and reading the text data
# URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
# urlopen(URL)
with open('input.txt', 'r', encoding='utf-8') as file:
    TEXT = file.read()

# Tokenization / Encoder & Decoder
CHARS = sorted(list(set(TEXT)))
VOCAB_SIZE = len(CHARS)
STOI = {ch: i for i, ch in enumerate(CHARS)}
ITOS = {i: ch for i, ch in enumerate(CHARS)}

ENCODE = lambda s: [STOI[c] for c in s]
DECODE = lambda l: ''.join([ITOS[i] for i in l])

# Train and Test splits
DATA = torch.tensor(ENCODE(TEXT), dtype=torch.long)
N = int(0.9 * len(DATA))
TRAIN_DATA = DATA[:N]
VAL_DATA = DATA[N:]
