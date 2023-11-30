# multi_head_attention.py

"""Multi-Head Self Attention module for transformer model."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import N_EMBED, DROPOUT
from feed_forward import FeedForward
from rotatry_embeding import RotaryEmbedding


class AttentionHead(nn.Module):
    def __init__(self, embed_size, head_size):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(embed_size, embed_size)))
        self.dropout = nn.Dropout(DROPOUT)
        self.rotary_embed = RotaryEmbedding(dim=head_size)

    def forward(self, x):
        B, T, C = x.shape
        key = self.key(x)
        query = self.query(x)

        if self.rotary_embed is not None:
            query = self.rotary_embed.rotate_queries_or_keys(query)
            key = self.rotary_embed.rotate_queries_or_keys(key)
        
        # Linear Self-attention
        weights = query @ key.transpose(-2, -1) * C ** -0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        value = self.value(x)

        # sliding window self-attention
        output = weights @ value
        return output


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(N_EMBED, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBED, N_EMBED)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        output = torch.cat([head(x) for head in self.heads], dim=-1)
        output = self.proj(output)
        output = self.dropout(output)
        return output


class Block(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        head_size = embed_size // num_heads
        self.self_attention = MultiHeadAttention(num_heads, head_size)
        self.feed_forward = FeedForward(embed_size)
        self.layer_norm1 = nn.LayerNorm(N_EMBED)
        self.layer_norm2 = nn.LayerNorm(N_EMBED)

    def forward(self, x):
        x = x + self.self_attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x
