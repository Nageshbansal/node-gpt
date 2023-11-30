# multi_head_attention.py

"""Multi-Head Self Attention module for transformer model."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from feed_forward import FeedForward
from rotatry_embeding import RotaryEmbedding


class AttentionHead(nn.Module):
    def __init__(self, embed_size, head_size, n_embd, dropout):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(embed_size, embed_size)))
        self.dropout = nn.Dropout(dropout)
        self.rotary_embed = RotaryEmbedding(dim=head_size)

    def forward(self, x):
        B, T, C = x.shape
        key = self.key(x)
        query = self.query(x)

        if self.rotary_embed is not None:
            query = self.rotary_embed.rotary_embedding(query)
            key = self.rotary_embed.rotary_embedding(key)
        
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
    def __init__(self, num_heads, head_size, dropout, n_embd):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(n_embd, head_size, n_embd, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = torch.cat([head(x) for head in self.heads], dim=-1)
        output = self.proj(output)
        output = self.dropout(output)
        return output


class Block(nn.Module):
    def __init__(self, embed_size, num_heads, dropout, n_embd):
        super().__init__()
        head_size = embed_size // num_heads
        self.self_attention = MultiHeadAttention(num_heads, head_size, dropout, n_embd)
        self.feed_forward = FeedForward(embed_size)
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.layer_norm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.self_attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x
