# main.py

"""Training and evaluation loop for the transformer model."""

from gpt import Transformer, TransformerConfig
import torch.optim as optim
import torch
from data_loading import get_batch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import os
import numpy as np
import tempfile

out_dir = 'out'
eval_interval = 2000
eval_iters = 200
always_save_checkpoint = True

# data
batch_size = 12
block_size = 16
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
# adamw optimizer
learning_rate =1e-3 # max learning rate
max_iters = 500

backend = 'gloo' # 'nccl', 'gloo', etc.
# system
device = 'cuda' 
ddp = True

# set rank for ddp
os.environ['RANK'] = '1'
os.environ['WORLD_SIZE'] = '0'

def set_process_mode():
    ok = False
    if ddp:
        ddp_rank = int(os.environ['RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend=backend, rank=ddp_rank, world_size=ddp_world_size)
        ok = True
    return ok

# load the dataset and set the process mode
ok = set_process_mode()
if ok:
    print("process mode : DDP")

def get_batch(split):
    """Generate a small batch of data for training or validation."""
    data = TRAIN_DATA if split == 'train' else VAL_DATA
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
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

def cleanup():
    dist.destroy_process_group()


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

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=False, vocab_size=None, dropout=dropout)


"""intiliazing the model"""
gptconf = TransformerConfig(**model_args)

if ddp:
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    ddp_local_rank = os.environ.get(['RANK'])
    model = Transformer(gptconf).to(ddp_local_rank)
    model = DDP(model, device_ids=[ddp_local_rank])
    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"

    if ddp_local_rank == 0:
        torch.save(model.state_dict(), CHECKPOINT_PATH)

    dist.barrier()

    map_location = {'cuda:%d' % 0: 'cuda:%d' % ddp_local_rank}
    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location))

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    for iter in range(max_iters):
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
    if ddp_local_rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()
