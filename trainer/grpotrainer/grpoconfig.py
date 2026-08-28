from typing_extensions import Literal

import torch
from dataclasses import dataclass

@dataclass
class GRPOConfig:
    # MISC
    total_samples:int
    current_example:int
    global_example:int
    test_train_ratio: float = 0.01
    min_lr_ratio:float = 0.8
    max_test_rows:int = 10000
    label_idx:int = -100
    
    
    # LEARNING PARAMETERS 
    batch_size: int = 2
    grad_accum_steps: int = 16
    warmup_steps_ratio:float = 0.10 # 10 % of total steps
    checkpoint_dir:str = "checkpoints"
    resume:bool = False
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
    
    
    # DISTRIBUTED
    distributed: str = "none"
    ddp_backend: str = "nccl"
    
    # Passed straight through to DistributedDataParallel. Only set this
    # to True if some parameters genuinely don't receive gradients on
    # every forward pass (e.g. certain MoE / routing architectures) -
    # it disables an optimization and slows every step down otherwise.
    ddp_find_unused_parameters: bool = False
    
    # Timeout (seconds) for collective NCCL/Gloo ops. The default
    # torch.distributed timeout (30 min) is usually fine, but slow
    # checkpoint saves or big eval sets on rank 0 can occasionally
    # trip the watchdog on other ranks, so it's exposed here.
    ddp_timeout_seconds: int = 1800
    ddp_sync_cooldown: bool = True # Enforce CoolDown for each distributed GPU