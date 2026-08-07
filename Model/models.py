import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.layers import Block, Config, RMSNorm

# BASE ARCHITECTURE: Attention with Linear Biases
class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model) # Tied Head + Embedding Layer
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layer)])
        self.head_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.final_norm = RMSNorm(config.d_model)   
        
        # Tie the weights
        self.head_proj.weight = self.embeddings.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, X: torch.Tensor, attention_mask=None):
        
        X = self.embeddings(X) # [B, L] --> [B, L, D]
        
        for block in self.blocks:
            X = block(X)
        
        X = self.final_norm(X)
        output = self.head_proj(X)
        
        return output # [B, L, VOCAB_SIZE]
    
        
# TO-DO
# [2] Padding Handling
# [3] Add Option for KV Cache for faster inference
# [4] Handle Long Generation: 30K