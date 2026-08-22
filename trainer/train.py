import argparse
from math import lgamma
import os

import yaml
from datasets import IterableDataset, load_dataset, load_dataset_builder
from datasets.arrow_dataset import Dataset
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# from Trainer.pretrainer import PreTrainer
from transformers import AutoTokenizer

from Model.layers import Config
from Model.models import Model
from trainer import SFTConfig, SFTTrainer, preprocess_rft, process_ift_dataset

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
        self.training_name = f"{args.training_name}_{args.pipeline}"
        self.console = Console()
        self.train(args)

    def train(self, args) -> None:

        config_path = os.path.join(BASE_DIR, "config.yaml")
        with open(config_path, "r") as file:
            data = yaml.safe_load(file)

        # Load Tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("Se00n00/TinyLM-2")
        match args.model:
            case "Alibi":
                model = Model(Config(vocab_size=len(tokenizer)))

            case _:
                model = Model(Config(vocab_size=len(tokenizer)))
        
        
        
        self._print_logo()
        
        os.makedirs(f"{args.checkpoint_dir}/{self.training_name}", exist_ok=True)
        for pipeline in data["pipeline"]:
            for dataset in data["dataset"][pipeline]:
                
                # PreLook the total samples of dataset and determine train and test samples
                builder = load_dataset_builder(dataset["base"], dataset["subset"])
                total_samples = builder.info.splits[dataset["split"]].num_examples
                total_samples = 1000
                training_samples = (
                    args.dataset_limit
                    if args.dataset_limit and dataset.get("limit") is None
                    else dataset.get("limit")
                    if dataset.get("limit")
                    else args.dataset_limit
                )
                test_train_ratio = (
                    0.01
                    if args.validation_dataset_limit is None
                    else args.validation_dataset_limit / total_samples
                )
                
                self.console.print(
                    f"\t.\n\t \\__[bold dim white]PIPELINE:[/] {pipeline}\n"
                    f"\t.\n\t \\__[bold dim white]DATASET:[/] {dataset['base']}\n"
                    f"\t.\n\t \\__[bold dim white]TOTAL SAMPLES:[/] {total_samples}\n"
                    f"\t.\n\t \\__[bold dim white]DOWNGRADED SAMPLES:[/] {training_samples}\n"
                    f"\t.\n\t \\__[bold dim white]TRAIN TEST RATIO:[/] {test_train_ratio}\n"
                    f"\t.\n\t \\__[bold dim white]TEST EXAMPLES:[/] {int(test_train_ratio * total_samples)}\n"
                )
                
                # Initiallize Trainer
                config = SFTConfig(
                    total_samples=training_samples,
                    batch_size=args.batch_size,
                    grad_accum_steps=args.grad_accum_steps,
                    resume=args.resume,
                    resum_same_dataset=args.resum_same_dataset,
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
                
                resume_state = SFTTrainer.peek_checkpoint(config, self.training_name)
                row_offset = resume_state["current_example"] if resume_state else 0
                starting_current_example = row_offset if args.resum_same_dataset else 0
        
                if resume_state is not None:
                    print(
                        f"[Resume] Found checkpoint at step {resume_state['step']} "
                        f"({row_offset} rows already consumed). "
                        f"{'Skipping ahead in the dataset.' if args.resum_same_dataset else 'Restarting dataset from the beginning.'}"
                    )
        
                ds = load_dataset(
                    dataset["base"], dataset["subset"], split=dataset["split"], streaming=args.stream_dataset
                )
                ds = self._change_template(ds, pipeline)
        
                
        
                if args.resum_same_dataset and row_offset > 0:
                    ds = ds.skip(row_offset)
        
                if args.dataset_limit:
                    ds.take(args.dataset_limit)
                    
                trainer_kwargs = {
                    "training_name": self.training_name,
                    "model": model,
                    "tokenizer": tokenizer,
                    "ds": ds,
                    "config": config,
                    "current_example": starting_current_example,
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
            
    def _change_template(self, dataset:Dataset | IterableDataset, pipeline:str):
        match pipeline:
            case "IFT":
                return IterableDataset.from_generator(
                    process_ift_dataset,
                    gen_kwargs={"dataset": dataset},
                )

            case "RIFT":
                return dataset.map(preprocess_rft, remove_columns=list(dataset.features.keys()))
            
            case "PT":
                return dataset

            case _:
                return dataset

    # -------------------------------------------------
    # CLI METHODS
    # -------------------------------------------------
    def _print_logo(self):
        f = Figlet(font="pagga")
        logo = f.renderText(" TinyLM Trainer")
        self.console.print("\n")
        self.console.print(logo)
        
    def _parse_arguments(self):
        parser = argparse.ArgumentParser(description="CLI TinyLM Trainer")

        parser.add_argument("--training_name", type=str)
        parser.add_argument("--model", type=str, choices=["Alibi"], default="Alibi")

        # LEARNING PARAMETERS
        parser.add_argument(
            "--batch_size",
            type=int,
            default=1,
            help="Micro-batch size",
        )
        parser.add_argument(
            "--grad_accum_steps",
            type=int,
            default=4,
            help="Gradient accumulation steps",
        )
        parser.add_argument(
            "--max_steps", type=int, default=20000, help="Total training steps"
        )
        parser.add_argument(
            "--complete_data",
            type=bool,
            default=False,
            help="Train on Complete Dataset ?",
        )
        parser.add_argument(
            "--warmup_steps_ratio", type=float, default=0.10, help="LR warmup steps"
        )
        parser.add_argument(
            "--checkpoint_dir",
            type=str,
            default="checkpoints",
            help="Directory to save model checkpoints",
        )
        parser.add_argument(
            "--resume",
            type=str,
            default=None,
            help="Path to checkpoint to resume training from (or 'auto' to auto-detect best checkpoint)",
        )
        parser.add_argument(
            "--resum_same_dataset",
            type=bool,
            default=False,
            help="Path to checkpoint to resume training from (or 'auto' to auto-detect best checkpoint)",
        )
        parser.add_argument(
            "--dataset_limit",
            type=int,
            default=None,
            help="Path to checkpoint to resume training from (or 'auto' to auto-detect best checkpoint)",
        )
        parser.add_argument(
            "--validation_dataset_limit",
            type=int,
            default=None,
            help="Path to checkpoint to resume training from (or 'auto' to auto-detect best checkpoint)",
        )
        parser.add_argument(
            "--stream_dataset",
            type=bool,
            default=False,
            help="train with streaming dataset ?",
        )
        parser.add_argument(
            "--learning_rate", type=float, default=3e-4, help="Max learning rate"
        )
        

        # LOGGING & EVALUATION
        parser.add_argument(
            "--eval_interval", type=int, default=2000, help="Steps between evaluations"
        )
        parser.add_argument(
            "--log_interval",
            type=int,
            default=20,
            help="Steps between monitor model internals",
        )

        # MODEL
        parser.add_argument(
            "--max_seq_len", type=int, default=512, help="Maximum Sequence length"
        )

        # MEMORY MANAGEMENT
        parser.add_argument(
            "--vram_limit_mb",
            type=int,
            default=30000,
            help="Target upper limit of VRAM usage in MB",
        )
        parser.add_argument(
            "--max_temp",
            type=int,
            default=75,
            help="GPU Temperature threshold to trigger cooldown in °C",
        )
        parser.add_argument(
            "--cooldown_temp",
            type=int,
            default=60,
            help="Target GPU Temperature to cool down to in °C",
        )

        # LEARNING PARAMETERS
        parser.add_argument(
            "--disable_amp",
            action="store_true",
            help="Disable automatic mixed precision (AMP)",
        )
        parser.add_argument(
            "--gradient_checkpointing",
            action="store_true",
            help="Start training with gradient checkpointing enabled",
        )
        parser.add_argument(
            "--weight_decay", type=float, default=0.1, help="Weight Decay rate"
        )
        parser.add_argument(
            "--pipeline",
            type=str,
            choices=[
                "PT",
                "IFT",
                "TC",  # Tool Calling
                "RIFT",  # Reasoning IFT
                "RTC",  # Reasoning Tool calling
            ],  # Pre-training, Instruction Finetunning, Preference Fine-Tunning
            default="PT",
            help="Pipeline process: pt, it,..",
        )
        # Distributed
        parser.add_argument(
            "--distributed",
            type=str,
            default="none",
            choices=["none", "ddp"],
            help="'ddp' enables multi-GPU training. If launched via `torchrun`, "
            "this is mostly informational - torchrun already sets the env vars "
            "SFTTrainer auto-detects. If launched as a plain `python train.py` "
            "with --distributed ddp and --world_size > 1, this script instead "
            "spawns the processes itself via SFTTrainer.launch().",
        )
        parser.add_argument(
            "--backend", type=str, default="nccl", choices=["nccl", "gloo"]
        )
        parser.add_argument(
            "--world_size",
            type=int,
            default=1,
            help="Only used for the self-spawning (`mp.spawn`) path. When "
            "launching with `torchrun --nproc_per_node=N`, N is what controls "
            "world size instead - leave this at 1.",
        )

        return parser.parse_args()


if __name__ == "__main__":
    train = Train()
