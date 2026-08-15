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
        "--learning_rate", type=float, default=3e-4, help="Max learning rate"
    )

    # LOGGING & EVALUATION
    parser.add_argument(
        "--eval_interval", type=int, default=200, help="Steps between evaluations"
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

EXAMPLE_DATASET = {"base": "Se00n00/FineWeb-1B", "subset": None, "split": "train"}
if __name__ == "__main__":
    args = parse_arguments()

    builder = load_dataset_builder(EXAMPLE_DATASET["base"])
    total_samples = builder.info.splits[EXAMPLE_DATASET["split"]].num_examples
    tokenizer = AutoTokenizer.from_pretrained("Se00n00/TinyLM-2")

    match args.model:
        case "Alibi":
            model = Model(Config(vocab_size=len(tokenizer)))

        case _:
            model = Model(Config(vocab_size=len(tokenizer)))

    config = SFTConfig(
        total_samples=1200,
        batch_size=args.batch_size,
        resume=args.resume,
        learning_rate=args.learning_rate,
        logging_steps=args.eval_interval,
        eval_steps=args.eval_interval,
        max_length=args.max_seq_len
    )
    os.makedirs(f"{args.checkpoint_dir}/{args.training_name}_{args.pipeline}", exist_ok=True)

    trainer = SFTTrainer(
        training_name=f"{args.training_name}_{args.pipeline}",
        model=model,
        tokenizer=tokenizer,
        ds=load_dataset(
            EXAMPLE_DATASET["base"], split=EXAMPLE_DATASET["split"], streaming=True
        ).take(1200),
        config=config,
    )

    trainer.train()
