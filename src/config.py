"""Configuration file for transformer model."""

import torch

# Hyperparameters
BATCH_SIZE = 32
BLOCK_SIZE = 8
MAX_ITERS = 5000
EVAL_INTERVAL = 500
LEARNING_RATE = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_ITERS = 200
N_EMBED = 512
N_HEAD = 8
N_LAYER = 6
HEAD_SIZE = 16
DROPOUT = 0.2
MAX_SEQ_LEN = 768
