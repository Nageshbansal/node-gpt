from torch import nn
import torch
from torch.nn import functional as F
from multi_head_attention import Block
from dataclasses import dataclass

@dataclass
class TransformerConfig:
    BATCH_SIZE: int = 32
    BLOCK_SIZE: int = 8
    MAX_ITERS: int = 5000
    EVAL_INTERVAL: int = 500
    LEARNING_RATE: float = 3e-4
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    EVAL_ITERS: int = 200
    N_EMBED: int = 512
    N_HEAD: int = 8
    N_LAYER: int = 6
    HEAD_SIZE: int = 16
    DROPOUT: float = 0.2
    MAX_SEQ_LEN: int = 768
    BIAS: bool = False
    VOCAB_SIZE: int = 65
    
    @classmethod
    def from_dict(cls, config_dict):
        # Convert keys to uppercase
        config_dict_upper = {key.upper(): value for key, value in config_dict.items()}
        return cls(**config_dict_upper)


class Transformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.config = config
        self.tokenizer = nn.Embedding(config.VOCAB_SIZE, config.N_EMBED)
        self.position_embed = nn.Embedding(config.BLOCK_SIZE, config.N_EMBED)
        self.drop = nn.Dropout(config.DROPOUT)
        self.blocks = nn.Sequential(*[Block(config.N_EMBED, num_heads=config.N_HEAD, dropout=config.DROPOUT, n_embd=config.N_EMBED) for i in range(config.N_LAYER)])
        self.ln_f = nn.LayerNorm(config.N_EMBED, bias=config.BIAS)
        # self.sa_head = Head(N_EMBED)
        # self.sa_heads = MultiHeadAttention(4, N_EMBED//4)

        self.lm_head = nn.Linear(config.N_EMBED, config.VOCAB_SIZE, bias=False)
        # self.ffwd = FeedForward(N_EMBED)
        # self.freqs_complex = precompute_thetha_pos_frequencies(N_EMBED // N_HEAD, max_seq_len * 2, device=device )

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embd = self.tokenizer(idx)

        # position embedding
        # pos_embd = self.position_embed(torch.arange(T, device=device))
        # x = token_embd + pos_embd

        x = token_embd
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):

            idx_cond = idx[:, -self.config.BLOCK_SIZE:]
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx