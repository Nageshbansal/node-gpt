# main.py

"""Training and evaluation loop for the transformer model."""

from gpt import Transformer, TransformerConfig
import torch.optim as optim
import torch
from data_loading import get_batch
from data_prepration import DECODE
import numpy as np


# data
eval_interval = 2000
eval_iters = 200
always_save_checkpoint = True

# data
batch_size = 32
block_size = 8
# model
n_layer = 6
n_head = 8
n_embd = 512
dropout = 0.2
EVAL_ITERS = 200
# adamw optimizer
learning_rate = 1e-3 
max_iters = 500
# system
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_batch(split):
    """Generate a small batch of data for training or validation."""
    data = TRAIN_DATA if split == 'train' else VAL_DATA
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x =  torch.stack([data[i:i + block_size] for i in ix])
    y =  torch.stack([data[i+1:i +1+ block_size] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    """Estimate loss on the train and validation sets."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


with open('input.txt', 'r', encoding='utf-8') as file:
    TEXT = file.read()

# Tokenization / Encoder & Decoder
chars = sorted(list(set(TEXT)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Train and Test splits
DATA = torch.tensor(encode(TEXT), dtype=torch.long)
N = int(0.9 * len(DATA))
TRAIN_DATA = DATA[:N]
VAL_DATA = DATA[N:]

model_args = dict(N_LAYER=n_layer, N_HEAD=n_head, N_EMBED=n_embd, BLOCK_SIZE=block_size,
                  BIAS=False, VOCAB_SIZE=vocab_size, DROPOUT=dropout)


"""initliazing the model"""
gptconf = TransformerConfig(**model_args)
model = Transformer(gptconf)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # Every once in a while, evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(DECODE(model.generate(context, max_new_tokens=1000)[0].tolist()))
