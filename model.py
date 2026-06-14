import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from dataclasses import dataclass

class LayerNorm(nn.Module):
    def __init__(self, d_model:int):
        super().__init__()

        self.weights = nn.Parameter(torch.ones(d_model)) # requires_grad = True
        self.bias = nn.Parameter(torch.ones(d_model))

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        normallized = (X - X.mean())/X.std()
        return normallized * self.weights + self.bias

class RMSNorm(nn.Module):
    def __init__(self, d_model:int, epsilon=1e-8):
        super().__init__()

        self.epsilon = epsilon
        self.weights = nn.Parameter(torch.ones(d_model))

    def forward(self, X:torch.Tensor) ->torch.Tensor:
        normallized = X * torch.rsqrt(X.pow(2).mean(-1, keepdim=True) + self.epsilon)

        print(normallized.shape)
        return normallized.type_as(X) * self.weights

class FeedForward(nn.Module):
    def __init__(self, d_model:int, hidden_d:int|None = None, gated:bool = True):
        super().__init__()

        if not hidden_d:
            hidden_d = 4*d_model

        self.is_gated = gated

        self.up_proj = nn.Linear(d_model, hidden_d, bias = False)
        if gated:
            self.gate_proj = nn.Linear(d_model, hidden_d, bias= False)
        self.down_proj = nn.Linear(hidden_d, d_model, bias = False)
        self.dropout = nn.Dropout(0.5)


    def forward(self, X:torch.Tensor) -> torch.Tensor:
        if self.is_gated:
            temp = F.silu(self.up_proj(X)) * self.gate_proj(X)
        else:
            temp = F.silu(self.up_proj(X))

        return self.dropout(self.down_proj(temp))

class Attention(nn.Module):
    def __init__(self, d_model:int, num_heads:int, max_len:int, dropout_prob:float):
        super().__init__()

        assert d_model % num_heads == 0, "Number of Heads must be divisible to model dim"
        head_dim = d_model // num_heads
        self.num_heads, self.head_dim = num_heads, head_dim

        self.q_proj = nn.Linear(d_model, head_dim * num_heads, bias = False)
        self.k_proj = nn.Linear(d_model, head_dim * num_heads, bias = False)
        self.v_proj = nn.Linear(d_model, head_dim * num_heads, bias = False)
        self.o_proj = nn.Linear(d_model, d_model, bias = False)

        self.attention_dropout = nn.Dropout(dropout_prob)
        self.output_dropout = nn.Dropout(dropout_prob)

        self.register_buffer(
                "attention_mask",
                torch.triu(torch.full((1, 1, max_len, max_len), float("-inf")), diagonal = 1)
                )

    def forward(self, x:torch.Tensor):
        batch, seqlen, d_model = x.shape
        
        # [B, L, head_dim * num_heads] <-- [B, L, D_model]
        Q,K,V = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Reshape <-- View | [B, num_heads, L, head_dim] <-- [B, L, num_heads, head_dim] <-- [B, L, num_heads * head_dim]
        Q = Q.view(batch, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seqlen, self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(batch, seqlen, self.num_heads, self.head_dim).transpose(1,2)

        scores = torch.matmul(Q, K.transpose(2,3)) / math.sqrt(d_model)
        scores = scores + self.attention_mask[:,:,seqlen, seqlen]
        scores = F.softmax(scores, dim = -1)
        scores = self.attention_dropout(scores)
        output = torch.matmul(scores, V)

        # [B, L, D_model] <-- [B, L, num_heads, head_dim] <-- [B, num_heads, L, head_dim]
        output = output.transpose(1, 2).contiguous().view(batch, seqlen, -1)
        return self.output_dropout(self.o_proj(output))

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size:int, block_size:int, d_model:int):
        super().__init__()
        self.token_encoding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = nn.Embedding(block_size, d_model)

    def forward(self, X:torch.Tensor):

        # [B, L, d_model] <-- ([B, L, d_model]+[B, L, d_model]) <-- [B,L]
        return self.token_encoding(X) + self.position_encoding(X)

@dataclass
class Config:
    num_layer: int
    max_len: int
    vocab_size: int
    block_size: int

    d_model: int
    num_heads: int
    dropout_prob: float
    
    # Feedforward
    ff_hidden_d: int|None
    ff_gated:bool

    # RMS Norm
    norm_epsilon:float


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        
        self.norm1 = RMSNorm(config.d_model, config.norm_epsilon)
        self.norm2 = RMSNorm(config.d_model, config.norm_epsilon)
        self.attention = Attention(config.d_model, config.num_heads, config.max_len, config.dropout_prob)
        self.feedforward = FeedForward(config.d_model, config.ff_hidden_d, config.ff_gated)

    def forward(self, X:torch.Tensor):
        X = X + self.attention(self.norm1(X))
        X = X + self.feedforward(self.norm2(X))

        return X

class Transformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.embedding = EmbeddingLayer(config.vocab_size, config.block_size, config.d_model)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layer)])
        self.head_proj = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, X:torch.Tensor):
        X = self.embedding(X)
        
        for block in self.blocks:
            X = block(X)

        output = self.head_proj(X)
        
        return output, output[:,-1, :] # [B, L, vocab_size], [B, 1, vocab_size]
        
