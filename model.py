import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

import math
from typing import Optional

@dataclass
class Config:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    muliple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0) -> torch.Tensor:

    theta_numerator = torch.arange(0, head_dim, 2).float() #shape => (head_dim/2)
    theta = (1.0 / math.pow(theta, theta_numerator/head_dim)).to(device) #shape => (head_dim/2)
    m = torch.arange(seq_len, device=device) #shape => (seq_len)
    freqs = torch.outer(m, theta).float() #shape => (m, head_dim/2)

    return torch.polar(torch.ones_like(freqs), freqs) #shape => (m, head_dim/2)

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str) -> torch.Tensor:
    
    x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2)) #shape => (b, s, h, d_dim, 2)
    x_rotated = x * freqs_complex.unsqueeze(0).unsqueeze(2)
    x_rotated = torch.view_as_real(x_rotated)

    return x_rotated.reshape(x.shape).type_as(x).to(device)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) #shape => (batch, seq_length, d_model)

    def forward(self, x: torch.Tensor):
        return self.g * self._norm(x.float()).type_as(x) #shape => (batch, seq_len, d_model)






class Transformer(nn.Module):
    def __init__(self, args: Config) -> None:
        super().__init__()

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.freqs_complex = precompute_theta_pos_frequencies(args.dim//args.n_heads, args.max_seq_len*2, device=args.device)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EcncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor, start_pos: int) -> torch.Tensor:

        batch_size, seq_len = tokens.shape #shape => (batch, seq_len)
        assert seq_len == 1, 'must be one token at a time'

        x = self.tok_embeddings(tokens) #shape => (batch, seq_len, dim)

        freq_complex =  self.freqs_complex[start_pos:start_pos+seq_len]

        for layer in self.layers:
            x = layer(x, start_pos, freq_complex)

        x = self.norm(x)

        x = self.output(x)

        return x

        
