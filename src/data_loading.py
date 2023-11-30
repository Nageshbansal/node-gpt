# data_loading.py

"""Data loading functions for transformer model."""

import torch
from data_prepration import TRAIN_DATA, VAL_DATA
from config import BLOCK_SIZE, BATCH_SIZE, DEVICE, EVAL_ITERS


# Data loading
def get_batch(split):
    """Generate a small batch of data for training or validation."""
    data = TRAIN_DATA if split == 'train' else VAL_DATA
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

