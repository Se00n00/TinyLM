import argparse
from math import lgamma
import os

import yaml
from datasets import IterableDataset, load_dataset, load_dataset_builder
from datasets.arrow_dataset import Dataset
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from typing import Dict
# from Trainer.pretrainer import PreTrainer
from transformers import AutoTokenizer

from Model.layers import Config
from Model.models import Model
from trainer import SFTConfig, SFTTrainer, preprocess_rft, process_ift_dataset, preprocess_text_generation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PRE_DATASET = {
    "base": "HuggingFaceFW/fineweb-edu",
    "subset": "sample-10BT",
    "split": "train",
}
IFT_DATASET = {
    "base": "HuggingFaceH4/ultrachat_200k",
    "subset": None,
    "split": "train_sft",
}
RFT_DATASET = {
    "base": "Scale-or-Reason/general-reasoning-ift-pairs",
    "subset": None,
    "split": "reasoning_ift_pairs",
}

import time
from rich import print
from pyfiglet import Figlet
from rich.layout import Layout

import logging
from rich.align import Align
from rich.logging import RichHandler

class Train:
    def __init__(self) -> None:
        args = self._parse_arguments()
        self.training_name = args.training_name
        self.console = Console()
        self.train(args)

    def train(self, args) -> None:

        # 1. Prepare Training MetaData: Training Details (from/to checkpoints), dataset details
        os.makedirs(f"{args.checkpoint_dir}/{self.training_name}", exist_ok=True)
        config_path = os.path.join(BASE_DIR, "config.yaml")
        training_path = os.path.join(f"{args.checkpoint_dir}/{self.training_name}", "training.yaml")
        
        with open(config_path, "r") as file:
            data = yaml.safe_load(file)
        
        
        if not os.path.exists(training_path):
            open(training_path, "w").close()
        
        with open(training_path, "r+") as file:
            training_info = yaml.safe_load(file)
            
            if training_info is None or args.resume is None:
                training_info = self._initiallize_training_details(data, file)
              
        
        # 2. Load Tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("Se00n00/TinyLM-2")
        match args.model:
            case "Alibi":
                model = Model(Config(vocab_size=len(tokenizer)))

            case _:
                model = Model(Config(vocab_size=len(tokenizer)))
        
        
        
        self._print_logo()
        

        for pipeline in data["pipeline"]:
            for idx ,dataset in enumerate(data["dataset"][pipeline]):
                
                # Look Ahead the total samples of dataset and determine train and test samples
                builder = load_dataset_builder(dataset["base"], dataset["subset"])
                total_samples = builder.info.splits[dataset["split"]].num_examples
                
                new_total_samples = (
                    args.dataset_limit
                    if args.dataset_limit and dataset.get("limit") is None
                    else dataset.get("limit")
                    if dataset.get("limit")
                    else total_samples
                )
                test_train_ratio = (
                    0.01
                    if args.validation_dataset_limit is None
                    else args.validation_dataset_limit / total_samples
                )
                training_samples = int(new_total_samples* ( 100 - test_train_ratio))
                
                if  training_samples <= training_info['pipeline'][pipeline][idx]['trained'] or training_info['pipeline'][pipeline][idx]['completed']:
                    continue
                
                self.console.print(
                    f"\t.\n\t \\__[bold dim white]PIPELINE:[/] {pipeline}\n"
                    f"\t.\n\t \\__[bold dim white]DATASET:[/] {dataset['base']}\n"
                    f"\t.\n\t \\__[bold dim white]TOTAL SAMPLES:[/] {total_samples}\n"
                    f"\t.\n\t \\__[bold dim white]DOWNGRADED SAMPLES:[/] {training_samples}\n"
                    f"\t.\n\t \\__[bold dim white]TRAIN TEST RATIO:[/] {test_train_ratio}\n"
                    f"\t.\n\t \\__[bold dim white]TEST EXAMPLES:[/] {int(test_train_ratio * total_samples)}\n"
                )
                
                row_offset = training_info['pipeline'][pipeline][idx]['trained']
                ds = load_dataset(
                    dataset["base"], dataset["subset"], split=dataset["split"], streaming=args.stream_dataset
                )
                ds = self._change_template(ds, pipeline, dataset) 
        
                
        
                if row_offset > 0:
                    ds = ds.skip(row_offset)
        
                if dataset.get("limit", None):
                    ds.take(dataset.get("limit"))
                
                # Initiallize Trainer
                config = SFTConfig(
                    total_samples=new_total_samples,
                    current_example= row_offset,
                    global_example= training_info['global_current_example'],
                    batch_size=args.batch_size,
                    grad_accum_steps=args.grad_accum_steps,
                    resume=args.resume, # <--- TODO: 2 - Fix Re-caliberated arguments and other things
                    # resum_same_dataset=args.resum_same_dataset,
                    learning_rate=args.learning_rate,
                    logging_steps=args.log_interval,
                    eval_steps=args.eval_interval,
                    max_length=args.max_seq_len,
                    checkpoint_dir=args.checkpoint_dir,
                    distributed=args.distributed,
                    ddp_backend=args.backend,
                    test_train_ratio=test_train_ratio,
                    warmup_steps_ratio=args.warmup_steps_ratio,
                    vram_limit_mb=args.vram_limit_mb,
                    max_temp=args.max_temp,
                    cooldown_temp=args.cooldown_temp,
                    weight_decay=args.weight_decay,
                    gradient_checkpointing=args.gradient_checkpointing,
                )
                
                trainer_kwargs = {
                    "training_name": self.training_name,
                    "training_config": training_info,
                    "pipeline": pipeline,
                    "dataset_name": dataset['base'],
                    "pipeline"
                    "model": model,
                    "tokenizer": tokenizer,
                    "ds": ds,
                    "config": config
                }
        
                if (
                    args.distributed == "ddp"
                    and args.world_size > 1
                    and "RANK" not in os.environ
                ):
                    SFTTrainer.launch(args.world_size, **trainer_kwargs)
                else:
                    trainer = SFTTrainer(**trainer_kwargs)
                    trainer.train()
                
                args.resume = "auto" # Enable Auto-resumption
             
    def _change_template(self, dataset:Dataset | IterableDataset, pipeline:str, dataset_config:Dict):
        match pipeline:
            case "IFT":
                return ( 
                    IterableDataset.from_generator(
                        process_ift_dataset,
                        gen_kwargs={"dataset": dataset},
                    ) if isinstance(dataset, IterableDataset)
                    else Dataset.from_generator(
                        process_ift_dataset,
                        gen_kwargs={"dataset": dataset}
                    )
                )

            case "RFT":
                return dataset.map(preprocess_rft, remove_columns=list(dataset.features.keys()))
            
            # case "TC":
            #     pass
                
            # case "RTC":
            #     pass
            
            case "PT":
                return (
                    dataset.map(preprocess_text_generation, remove_columns=list(dataset.features.keys()), fn_kwargs={"text_column":dataset_config["text_column"]}) 
                    if isinstance(dataset, Dataset)
                    else dataset.map(preprocess_text_generation, remove_columns=list(dataset.features.keys()), gen_kwargs={"text_column":dataset_config["text_column"]}) 
                )

            case _: # Pre-training
                return (
                    dataset.map(preprocess_text_generation, remove_columns=list(dataset.features.keys()), fn_kwargs={"text_column":dataset_config["text_column"]}) 
                    if isinstance(dataset, Dataset)
                    else dataset.map(preprocess_text_generation, remove_columns=list(dataset.features.keys()), gen_kwargs={"text_column":dataset_config["text_column"]}) 
                )
    
    def _initiallize_training_details(self, data, file):
            initial_data = {
                "current_pipeline": data['pipeline'][0],
                "global_current_example": 0,
                "pipeline": {
                    pipeline_name: [
                        {
                            "dataset": d['base'],
                            "trained": 0,
                            "completed": False # Skips  Configuration complexity for skipping data
                        } for d in data['dataset'][pipeline_name]
                    ] for i, pipeline_name in enumerate(data['pipeline']) 
                }
            }
            
            file.seek(0)
            yaml.dump(
                initial_data,
                file,
                default_flow_style=False,
                sort_keys=False,
            )
            file.truncate()
            return initial_data
            
    # -------------------------------------------------
    # CLI METHODS
    # -------------------------------------------------
    def _print_logo(self):
        f = Figlet(font="pagga")
        logo = f.renderText(" TinyLM Trainer")
        self.console.print("\n")
        self.console.print(logo)
        
    def _parse_arguments(self):
        parser = argparse.ArgumentParser(
            description="TinyLM CLI Trainer – train language models with flexible configuration for single- or multi-GPU setups."
        )
    
        # ----------------------------------------------------------
        # GENERAL
        # ----------------------------------------------------------
        parser.add_argument(
            "--training_name",
            type=str,
            help="Unique name for this training run (used for logging, checkpoint naming, and experiment tracking).",
        )
        parser.add_argument(
            "--model",
            type=str,
            choices=["Alibi"],
            default="Alibi",
            help="Model architecture to train. Currently only 'Alibi' is supported.",
        )
    
        # ----------------------------------------------------------
        # LEARNING PARAMETERS
        # ----------------------------------------------------------
        parser.add_argument(
            "--batch_size",
            type=int,
            default=1,
            help="Micro-batch size per GPU (number of sequences processed in one forward/backward pass).",
        )
        parser.add_argument(
            "--grad_accum_steps",
            type=int,
            default=4,
            help="Number of micro-batches to accumulate gradients over before performing an optimizer step. "
                 "Effective batch size = batch_size × grad_accum_steps × world_size.",
        )
        parser.add_argument(
            "--warmup_steps_ratio",
            type=float,
            default=0.10,
            help="Fraction of total training steps used for learning-rate warmup (linear ramp from 0 to max LR). "
                 "Example: 0.10 means the first 10%% of steps are warmup.",
        )
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=3e-4,
            help="Peak (maximum) learning rate reached after warmup.",
        )
        parser.add_argument(
            "--weight_decay",
            type=float,
            default=0.1,
            help="Weight decay (L2 regularization) coefficient applied by the optimizer.",
        )
        parser.add_argument(
            "--disable_amp",
            action="store_true",
            help="Disable Automatic Mixed Precision (AMP). Training will run in full FP32 (slower, higher memory).",
        )
        parser.add_argument(
            "--gradient_checkpointing",
            action="store_true",
            help="Enable gradient checkpointing at the start of training to reduce activation memory "
                 "(trades compute for lower VRAM usage).",
        )
    
        # ----------------------------------------------------------
        # CHECKPOINTING & RESUME
        # ----------------------------------------------------------
        parser.add_argument(
            "--checkpoint_dir",
            type=str,
            default="checkpoints",
            help="Directory where model checkpoints will be saved.",
        )
        parser.add_argument(
            "--resume",
            type=bool,
            default=False,
            help="Resume Training from already trained model (in checkpoint_dir) ?"
        )
    
        # ----------------------------------------------------------
        # DATA
        # ----------------------------------------------------------
        parser.add_argument(
            "--validation_dataset_limit",
            type=int,
            default=None,
            help="Maximum number of examples to use from the validation set. "
                 "Useful for faster evaluation during development. None = use the full validation set.",
        )
        parser.add_argument(
            "--stream_dataset",
            type=bool,
            default=False,
            help="If True, load the training dataset in streaming mode (lower memory, no full dataset caching).",
        )
        parser.add_argument(
            "--max_seq_len",
            type=int,
            default=512,
            help="Maximum sequence length (in tokens) for both training and evaluation.",
        )
    
        # ----------------------------------------------------------
        # LOGGING & EVALUATION
        # ----------------------------------------------------------
        parser.add_argument(
            "--eval_interval",
            type=int,
            default=2000,
            help="Run evaluation on the validation set every N optimizer steps.",
        )
        parser.add_argument(
            "--log_interval",
            type=int,
            default=20,
            help="Log training metrics and monitor model internals every N optimizer steps.",
        )
    
        # ----------------------------------------------------------
        # MEMORY & HARDWARE MANAGEMENT
        # ----------------------------------------------------------
        parser.add_argument(
            "--vram_limit_mb",
            type=int,
            default=30000,
            help="Soft upper limit for VRAM usage in megabytes. The trainer may adjust settings to stay under this limit.",
        )
        parser.add_argument(
            "--max_temp",
            type=int,
            default=75,
            help="GPU temperature (°C) at which a cooldown pause is triggered.",
        )
        parser.add_argument(
            "--cooldown_temp",
            type=int,
            default=60,
            help="Target GPU temperature (°C) to reach during cooldown before resuming training.",
        )
    
        # ----------------------------------------------------------
        # DISTRIBUTED TRAINING
        # ----------------------------------------------------------
        parser.add_argument(
            "--distributed",
            type=str,
            default="none",
            choices=["none", "ddp"],
            help="'ddp' enables multi-GPU DistributedDataParallel training. "
                 "When launched via `torchrun`, this flag is mostly informational "
                 "(environment variables are already set). "
                 "When launched as a plain `python train.py` with --distributed ddp and --world_size > 1, "
                 "the script will spawn the processes itself.",
        )
        parser.add_argument(
            "--backend",
            type=str,
            default="nccl",
            choices=["nccl", "gloo"],
            help="Distributed communication backend. Prefer 'nccl' for multi-GPU NVIDIA setups; use 'gloo' for CPU or debugging.",
        )
        parser.add_argument(
            "--world_size",
            type=int,
            default=1,
            help="Number of processes / GPUs for the self-spawning (mp.spawn) path. "
                 "Ignored when launching with `torchrun --nproc_per_node=N` (N becomes the world size).",
        )
    
        return parser.parse_args()


if __name__ == "__main__":
    train = Train()
