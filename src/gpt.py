from torch import nn
import torch
from torch.nn import functional as F
from multi_head_attention import Block
from config import N_EMBED, BLOCK_SIZE, N_HEAD, N_LAYER
from data_prepration import VOCAB_SIZE


class Transformer(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBED)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBED)
        self.blocks = nn.Sequential(*[Block(N_EMBED, num_heads=N_HEAD) for i in range(N_LAYER)])
        # self.sa_head = Head(N_EMBED)
        # self.sa_heads = MultiHeadAttention(4, N_EMBED//4)
        self.ln_f = nn.LayerNorm(N_EMBED)
        self.lm_head = nn.Linear(N_EMBED, VOCAB_SIZE)
        # self.ffwd = FeedForward(N_EMBED)
        # self.freqs_complex = precompute_thetha_pos_frequencies(N_EMBED // N_HEAD, max_seq_len * 2, device=device )

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        token_embd = self.token_embedding_table(idx) # (B,T,C)
        # pos_embd = self.position_embedding_table(torch.arange(T, device=device))
        # x = token_embd + pos_embd
        # x = self.sa_heads(x)
        # x = self.ffwd(x)
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

            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx