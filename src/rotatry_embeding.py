import torch
from torch.cuda.amp import autocast
from torch import nn, einsum, broadcast_tensors
from einops import rearrange, repeat


def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')


@autocast(enabled=False)
def apply_rotary_emb(freqs, t, start_index=0, scale=1., seq_dim=-2):
    rot_dim, seq_len = freqs.shape[-1], t.shape[seq_dim]
    freqs = freqs[-seq_len:].to(t)
    end_index = start_index + rot_dim
    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t_left, t, t_right), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        theta=10000,
        learned_freq=False,
        seq_before_head_dim=False
    ):
        super().__init__()

        theta *= 1.0 ** (dim / (dim - 2))
        freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        self.cache = dict()
        self.freqs = nn.Parameter(freqs, requires_grad=learned_freq)

        self.learned_freq = learned_freq
        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer('scale', scale)

    def get_seq_pos(self, seq_len, device, dtype, offset=0):
        return (torch.arange(seq_len, device=device, dtype=dtype) + offset) / 1.0

    def rotary_embedding(self, t, seq_dim=None, offset=0, freq_seq_len=None):
        seq_dim = default(seq_dim, self.default_seq_dim)
        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]

        if exists(freq_seq_len):
            assert freq_seq_len >= seq_len
            seq_len = freq_seq_len

        freqs = self.forward(lambda: self.get_seq_pos(seq_len, device=device, dtype=dtype, offset=offset),
                             cache_key=f'freqs:{seq_len}|offset:{offset}')

        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')

        return apply_rotary_emb(freqs, t, seq_dim=seq_dim)

    @autocast(enabled=False)
    def forward(self, t, cache_key=None):
        should_cache = not self.learned_freq and exists(cache_key)

        if should_cache and cache_key in self.cache:
            return self.cache[cache_key]

        if callable(t):
            t = t()

        freqs = self.freqs

        freqs = einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r=2)

        if should_cache:
            self.cache[cache_key] = freqs

        return freqs

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def broadcast(tensors, dim=-1):
    broadcasted_tensors = broadcast_tensors(*tensors)
    return torch.cat(broadcasted_tensors, dim=dim)
