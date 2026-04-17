import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass



# CONFIG

@dataclass
class ModelConfig:
    num_hidden_layers: int = 6
    hidden_size: int = 384

    num_attention_heads: int = 6
    num_key_value_heads: int = 2
    head_dim: int = 64

    intermediate_size: int = 1536

    vocab_size: int = 50000
    max_seq_len: int = 1024

    rope_theta: float = 10000.0
    rope_scaling: float = 4.0

    eps: float = 1e-5



# RMSNorm

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.scale * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)



# RoPE (with scaling)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000, scale=1.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.scale = scale

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        B, T, H, D = x.shape

        t = torch.arange(T, device=x.device) / self.scale
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        cos = freqs.cos()[None, :, None, :]
        sin = freqs.sin()[None, :, None, :]

        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat([x1 * cos - x2 * sin,
                          x2 * cos + x1 * sin], dim=-1)



# ATTENTION

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        self.q = nn.Linear(config.hidden_size,
                           self.num_heads * self.head_dim)

        self.k = nn.Linear(config.hidden_size,
                           self.num_kv_heads * self.head_dim)

        self.v = nn.Linear(config.hidden_size,
                           self.num_kv_heads * self.head_dim)

        self.out = nn.Linear(self.num_heads * self.head_dim,
                             config.hidden_size)

        self.rope = RotaryEmbedding(self.head_dim,
                                    config.rope_theta,
                                    config.rope_scaling)

    def forward(self, x):
        B, T, _ = x.shape

        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_kv_heads, self.head_dim)

        q = self.rope(q)
        k = self.rope(k)

        repeat = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat, dim=2)
        v = v.repeat_interleave(repeat, dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out(out)



# MLP (SwiGLU)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size,
                             config.intermediate_size * 2)
        self.fc2 = nn.Linear(config.intermediate_size,
                             config.hidden_size)

    def forward(self, x):
        x1, x2 = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(x1 * F.silu(x2))



# BLOCK

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size)
        self.norm2 = RMSNorm(config.hidden_size)

        self.attn = Attention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + 0.5 * self.attn(self.norm1(x))
        x = x + 0.5 * self.mlp(self.norm2(x))
        return x



# MODEL

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embed = nn.Embedding(config.vocab_size,
                                  config.hidden_size)

        self.blocks = nn.ModuleList(
            [Block(config) for _ in range(config.num_hidden_layers)]
        )

        self.norm = RMSNorm(config.hidden_size)

        self.lm_head = nn.Linear(config.hidden_size,
                                 config.vocab_size,
                                 bias=False)

        # Apply custom weight initialization first
        self.apply(self._init_weights)

        # Tie weights after initialization
        self.lm_head.weight = self.embed.weight

    def _init_weights(self, module):
        # Use a consistent std for initial weights
        # The `len(self.blocks)` is essentially `config.num_hidden_layers`
        std = 0.02 / math.sqrt(2 * len(self.blocks))

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)
