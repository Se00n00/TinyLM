import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="TinyLLM Trainer")

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
        "--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--max_steps", type=int, default=20000, help="Total training steps"
    )
    parser.add_argument(
        "--complete_data", type=bool, default=False, help="Train on Complete Dataset ?"
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
        default=4000,
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
            "PFT",
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
    parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"])
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Only used for the self-spawning (`mp.spawn`) path. When "
        "launching with `torchrun --nproc_per_node=N`, N is what controls "
        "world size instead - leave this at 1.",
    )

    args = parser.parse_args()

    return args


LOCAL_DATAPATH = {
    "train": {
        "token_path": "Datasets/Pre_Training/dataset_train.bin",
    },
    "validation": {
        "token_path": "Datasets/Pre_Training/dataset_validation.bin",
    },
    "test": {
        "token_path": "Datasets/Pre_Training/dataset_test.bin",
    },
}

PRETRAINING_HUGGINGFACE_DATASET = [
    {
        "base": "HuggingFaceFW/fineweb",
        "subset": "sample-10BT",
        "split": "train",
    },
    {
        "base": "HuggingFaceFW/fineweb-edu",
        "subset": "sample-10BT",
        "split": "train",
    },
    {
        "base": "bigcode/the-stack-v2",
        "subset": "JSON",
        "split": "train",
    },
    {
        "base": "bigcode/the-stack-v2",
        "subset": "Shell",
        "split": "train",
    },
    {
        "base": "HuggingFaceTB/smollm-corpus",
        "subset": "cosmopedia-v2",
        "split": "train",
    },
    {
        "base": "bigcode/the-stack-v2",
        "subset": "API_Blueprint",
        "split": "train",
    },
    {
        "base": "bigcode/the-stack-v2",
        "subset": "Python",
        "split": "train",
    },
    {
        "base": "emozilla/pg19",
        "subset": None,
        "split": "train",
    },
]

from datasets import load_dataset, load_dataset_builder

# from Trainer.pretrainer import PreTrainer
from transformers import AutoTokenizer
import os
from Model.layers import Config
from Model.models import Model
from trainer import SFTConfig, SFTTrainer

EXAMPLE_DATASET = {"base": "HuggingFaceFW/fineweb-edu", "subset": "sample-10BT", "split": "train"}
if __name__ == "__main__":
    args = parse_arguments()
    training_name = f"{args.training_name}_{args.pipeline}"

    builder = load_dataset_builder(EXAMPLE_DATASET["base"])
    total_samples = builder.info.splits[EXAMPLE_DATASET["split"]].num_examples
    tokenizer = AutoTokenizer.from_pretrained("Se00n00/TinyLM-2")

    print("TOTAL SAMPLE OF DATASET: ", total_samples, "\n\n")
    match args.model:
        case "Alibi":
            model = Model(Config(vocab_size=len(tokenizer)))

        case _:
            model = Model(Config(vocab_size=len(tokenizer)))
    
    total_samples = total_samples if args.dataset_limit is None else args.dataset_limit
    test_train_ratio = 0.01 if args.validation_dataset_limit is None else args.validation_dataset_limit / total_samples
    print(f"TOTAL SAMPLE OF DATASET: {total_samples} | TRAIN TEST RATIO: {test_train_ratio} | TEST EXAMPLES: {test_train_ratio * total_samples}\n\n")
    config = SFTConfig(
        total_samples=total_samples ,
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
        test_train_ratio = test_train_ratio,
        vram_limit_mb=args.vram_limit_mb,
        max_temp=args.max_temp,
        cooldown_temp=args.cooldown_temp,
        weight_decay=args.weight_decay,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    
    os.makedirs(f"{args.checkpoint_dir}/{training_name}", exist_ok=True)
    
    # -----------------------------------------------------------------
    # RESUME: figure out how many rows to skip BEFORE building the
    # dataset. See the write-up below for why this has to happen here,
    # at the raw-dataset level, rather than inside SFTTrainer.
    # -----------------------------------------------------------------
    resume_state = SFTTrainer.peek_checkpoint(config, training_name)
    row_offset = resume_state["current_example"] if resume_state else 0
    starting_current_example = row_offset if args.resum_same_dataset else 0
    
    if resume_state is not None:
        print(
            f"[Resume] Found checkpoint at step {resume_state['step']} "
            f"({row_offset} rows already consumed). "
            f"{'Skipping ahead in the dataset.' if args.resum_same_dataset else 'Restarting dataset from the beginning.'}"
        )
    
    ds = load_dataset(
        EXAMPLE_DATASET["base"], split=EXAMPLE_DATASET["split"], streaming=True
    )
    if args.resum_same_dataset and row_offset > 0:
        ds = ds.skip(row_offset)
    
    if args.dataset_limit:
        ds.take(args.dataset_limit)
    trainer_kwargs = dict(
        training_name=training_name,
        model=model,
        tokenizer=tokenizer,
        ds=ds,
        config=config,
        current_example=starting_current_example,
    )
    
    # If launched with multiple processes via mp.spawn without torchrun
    if args.distributed == "ddp" and args.world_size > 1 and "RANK" not in os.environ:
        SFTTrainer.launch(args.world_size, **trainer_kwargs)
    else:
        trainer = SFTTrainer(**trainer_kwargs)
        trainer.train()