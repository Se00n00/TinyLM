# Layers:
#   EmbeddingLayer
#   Normallization Layers: LayerNorm, RMSNorm
#   Attention: MHA
#   FeedForward
# Block

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, epsilon=1e-8):
        super().__init__()

        self.epsilon = epsilon
        self.weights = nn.Parameter(torch.ones(d_model))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        normallized = X * torch.rsqrt(X.pow(2).mean(-1, keepdim=True) + self.epsilon)

        return normallized.type_as(X) * self.weights


class Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_prob: float):
        super().__init__()
        
        assert d_model % num_heads == 0, (
            "Number of Heads must be divisible to model dim"
        )
        head_dim = d_model // num_heads
        self.num_heads, self.head_dim = num_heads, head_dim

        self.q_proj = nn.Linear(d_model, head_dim * num_heads, bias=False)
        self.k_proj = nn.Linear(d_model, head_dim * num_heads, bias=False)
        self.v_proj = nn.Linear(d_model, head_dim * num_heads, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.output_dropout = nn.Dropout(dropout_prob)
        self.attention_dropout = nn.Dropout(dropout_prob)
        
        self.register_buffer(
            "alibi_slopes",
            self.get_alibi_slopes(num_heads).view(1, num_heads, 1, 1),
            persistent=False,
        )

    def get_alibi_slopes(self, num_heads: int):
        """
        Returns the ALiBi slopes from the paper.
        """
    
        def get_slopes_power_of_2(n):
            start = 2 ** (-2 ** -(math.log2(n) - 3))
            ratio = start
            return torch.tensor(
                [start * ratio ** i for i in range(n)],
                dtype=torch.float32,
            )
    
        if math.log2(num_heads).is_integer():
            return get_slopes_power_of_2(num_heads)
    
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    
        return torch.cat([
            get_slopes_power_of_2(closest_power_of_2),
            self.get_alibi_slopes(2 * closest_power_of_2)[0::2][: num_heads - closest_power_of_2],
        ])


    def forward(self, x: torch.Tensor):
        batch, seqlen, d_model = x.shape

        # [B, L, head_dim * num_heads] <-- [B, L, D_model]
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Reshape <-- View | [B, num_heads, L, head_dim] <-- [B, L, num_heads, head_dim] <-- [B, L, num_heads * head_dim]
        Q = Q.view(batch, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seqlen, self.num_heads, self.head_dim).transpose(1, 2)

        # ---------------- ALiBi ----------------

        pos = torch.arange(seqlen, device=x.device)

        # distance[i,j] = max(i-j, 0)
        # | 0 0 0 0 0 0 0 0 |
        # | 1 0 0 0 0 0 0 0 |
        # | 2 1 0 0 0 0 0 0 |
        # | 3 2 1 0 0 0 0 0 |
        # | 4 3 2 1 0 0 0 0 |
        # | . 4 3 2 1 0 0 0 |
        # | . . 4 3 2 1 0 0 |
        # | P . . 4 3 2 1 0 |
        
        distance = (pos[:, None] - pos[None, :]).clamp(min=0)

        # [1, H, L, L]
        alibi = -self.alibi_slopes * distance
        
        # CAUSAL MASK
        causal_mask = torch.tril(torch.ones(seqlen, seqlen)).bool()
        
        combined_mask = alibi.clone()
        combined_mask = combined_mask.to(x.device).masked_fill(~causal_mask.to(x.device), float("-inf"))
        # ---------------------------------------

        
        output = F.scaled_dot_product_attention(
           Q,
           K,
           V,
           attn_mask=combined_mask,
           dropout_p=self.attention_dropout.p if self.training else 0.0,
           is_causal=False,
        )
        # scores = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.head_dim)
        # scores = scores + self.attention_mask[:, :, :seqlen, :seqlen]
        # scores = F.softmax(scores, dim=-1)
        # scores = self.attention_dropout(scores)
        # output = torch.matmul(scores, V)

        # [B, L, D_model] <-- [B, L, num_heads, head_dim] <-- [B, num_heads, L, head_dim]
        output = output.transpose(1, 2).contiguous().view(batch, seqlen, -1)
        return self.output_dropout(self.o_proj(output))


class FeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_d: int | None = None,
        gated: bool = True,
        dropout_prob=0.5,
    ):
        super().__init__()

        if not hidden_d:
            hidden_d = 4 * d_model

        self.is_gated = gated

        self.up_proj = nn.Linear(d_model, hidden_d, bias=False)
        if gated:
            self.gate_proj = nn.Linear(d_model, hidden_d, bias=False)
        self.down_proj = nn.Linear(hidden_d, d_model, bias=False)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.is_gated:
            temp = F.silu(self.gate_proj(X)) * self.up_proj(X)
        else:
            temp = F.silu(self.up_proj(X))

        return self.dropout(self.down_proj(temp))


class MoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        top_k: int = 2,
        num_of_experts: int = 8,
        hidden_d: int | None = None,
        gated: bool = True,
        dropout_prob=0.5,
    ) -> None:
        super().__init__()
        assert top_k <=num_of_experts, "Top K experts must be less or equal to number of experts"
        self.top_k = top_k
        
        self.experts = nn.ModuleList(
            [
                FeedForward(d_model, hidden_d, gated, dropout_prob)
                for _ in range(num_of_experts)
            ]
        )
        self.W_expert_gate = nn.Linear(d_model, num_of_experts)
    
    def forward(self, X:torch.Tensor) -> torch.Tensor:
        router = self.W_expert_gate(X) # [B, L, EXPERTS]
        top_k = torch.topk(router, self.top_k, dim=-1) # values, indices: [B, L, TOP_K]
        
        router_logits = F.softmax(top_k.values, dim=-1)
    
    def _get_top_k(self, router:torch.Tensor) -> tuple[torch.Tensor, list]:
        
        return torch.tensor(), []


@dataclass
class Config:
    num_layer: int = 8
    vocab_size: int = 30000

    d_model: int = 768
    num_heads: int = 12
    dropout_prob: float = 0.1

    # Feedforward
    ff_hidden_d: int | None = 1365
    ff_gated: bool = True

    # RMS Norm
    norm_epsilon: float = 1e-8


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.norm1 = RMSNorm(config.d_model, config.norm_epsilon)
        self.norm2 = RMSNorm(config.d_model, config.norm_epsilon)
        self.attention = Attention(
            config.d_model, config.num_heads, config.dropout_prob
        )
        self.feedforward = FeedForward(
            config.d_model, config.ff_hidden_d, config.ff_gated, config.dropout_prob
        )

    def forward(self, X: torch.Tensor):
        X = X + self.attention(self.norm1(X))
        X = X + self.feedforward(self.norm2(X))

        return X
