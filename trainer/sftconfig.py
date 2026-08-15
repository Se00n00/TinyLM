from typing_extensions import Literal

import torch
from dataclasses import dataclass

@dataclass
class SFTConfig:
    # MISC
    total_samples:int
    test_train_ratio: float = 0.01
    min_lr_ratio = 0.8
    max_test_rows = 10000
    label_idx = -100
    
    # LEARNING PARAMETERS 
    batch_size: int = 2
    grad_accum_steps: int = 16
    warmup_steps_ratio:float = 0.10 # 10 % of total steps
    checkpoint_dir:str = "checkpoints"
    resume:str | None = None
    resum_same_dataset:bool = False
    learning_rate: float = 2e-5
    
    # LOGGING & EVALUATION
    logging_steps: int = 10
    eval_steps:int = 200
    
    # MODEL PARAMTERS
    max_length: int = 512
    ptdtype = torch.float16
    
    # MEMORY MANAGEMENT
    vram_limit_mb:int = 4000
    max_temp:int = 75
    cooldown_temp:int = 60
    
    # TRAINING
    gradient_checkpointing: bool = True
    weight_decay:float = 0.1