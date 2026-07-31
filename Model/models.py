import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.layers import Block, Config


class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model) # Tied Head + Embedding Layer
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layer)])
        self.head_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
                
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
    
    def forward(self, X: torch.Tensor):
        X = self.embeddings(X) # [B, L] --> [B, L, D]

        for block in self.blocks:
            X = block(X)

        output = self.head_proj(X)
        
        return output # [B, L, VOCAB_SIZE]