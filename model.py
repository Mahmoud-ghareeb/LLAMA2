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
    norm_eps: float = 1e-6

    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0) -> torch.Tensor:

    theta_numerator = torch.arange(0, head_dim, 2).float() #shape => (head_dim/2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device) #shape => (head_dim/2)
    m = torch.arange(seq_len, device=device) #shape => (seq_len)
    freqs = torch.outer(m, theta).float() #shape => (m, head_dim/2)

    return torch.polar(torch.ones_like(freqs), freqs) #shape => (m, head_dim/2)

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str) -> torch.Tensor:
    
    x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2)) #shape => (b, s, h, d_dim, 2)
    x_rotated = x_complex.to(device) * freqs_complex.unsqueeze(0).unsqueeze(2).to(device)
    x_rotated = torch.view_as_real(x_rotated)

    return x_rotated.reshape(*x.shape).type_as(x).to(device)

def repeat_kv(x: torch.Tensor, n: int) -> torch.Tensor:
    batch, seq_len, n_kv_heads, dim = x.shape
    if n == 1:
        return x
    else:
        return (
            x[:, :, :, None, :]
            .expand(batch, seq_len, n_kv_heads, n, dim)
            .reshape(batch, seq_len, n_kv_heads * n, dim)
        )

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) #shape => (batch, seq_length, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.g * self._norm(x.float()).type_as(x) #shape => (batch, seq_len, d_model)


class GroupedQueryAttentionWithKVCache(nn.Module):
    def __init__(self, args: Config) -> None:
        super().__init__()

        self.n_heads = args.n_heads
        self.kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads

        self.n_rep = self.n_heads // self.kv_heads
        self.head_dim = args.dim // self.n_heads

        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, self.kv_heads*self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.kv_heads*self.head_dim, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.kv_heads, self.head_dim))



    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        #split the d_diminsion into n_heads
        xq = xq.view(batch, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch, seq_len, self.n_heads, self.head_dim)
        xv = xv.view(batch, seq_len, self.n_heads, self.head_dim)

        #add RoPE to Query and Keys
        xq = apply_rotary_embeddings(xq, freqs_complex, x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, x.device)

        #kv_cache
        self.cache_k[:batch, start_pos:start_pos+seq_len] = xk
        self.cache_v[:batch, start_pos:start_pos+seq_len] = xv

        keys = self.cache_k[:batch, 0:start_pos+seq_len]
        values = self.cache_v[:batch, 0:start_pos+seq_len]

        #repeat keys and values as much as the ratio between the n_heads and kv_heads
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        scores = F.softmax((torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)).float(), dim=-1).type_as(xq)
        out = torch.matmul(scores, values)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.wo(out)



class FFSwiGLU(nn.Module):
    def __init__(self, args: Config) -> None:
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)

        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)

        hidden_dim = args.muliple_of * ((hidden_dim + args.muliple_of - 1) // args.muliple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class EcncoderBlock(nn.Module):
    def __init__(self, args: Config) -> None:
        super().__init__()

        self.rms1 = RMSNorm(args.dim, args.norm_eps)

        self.self_attention = GroupedQueryAttentionWithKVCache(args)

        self.rms2 = RMSNorm(args.dim, args.norm_eps)

        self.ffn = FFSwiGLU(args)

    def forward(self, x: torch.Tensor, start_pos: int, freq_complex: torch.Tensor) -> torch.Tensor:

        x = self.rms1(x)
        attn = self.self_attention(x, start_pos, freq_complex)
        x = x + attn
        rms = self.rms2(x)
        y = self.ffn(rms)
        x = x + y

        return x



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
    
if __name__ == '__main__':
    x = torch.randint(0, 10, (1, 1))
    con = Config()
    con.dim = 512
    con.max_batch_size = 1
    con.max_seq_len = 10
    con.device = 'cuda'
    con.vocab_size = 100
    print(Transformer(con).to()(x, 0))
        
