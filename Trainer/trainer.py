import argparse
from Trainer.initiallize import initialize
def parse_arguments():
    parser = argparse.ArgumentParser(description="TinyLLM Trainer")

    parser.add_argument("--training_name", type=str)
    parser.add_argument("--model", type=str, choices=["Alibi"], default="Alibi")
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
        "--max_seq_len", type=int, default=512, help="Maximum Sequence length"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0, help="Max learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.1, help="Weight Decay rate"
    )
    parser.add_argument(
        "--max_steps", type=int, default=20000, help="Total training steps"
    )
    parser.add_argument(
        "--complete_data", type=bool, default=False, help="Train on Complete Dataset ?"
    )
    parser.add_argument("--warmup_steps", type=int, default=2000, help="LR warmup steps")
    parser.add_argument(
        "--eval_interval", type=int, default=200, help="Steps between evaluations"
    )
    parser.add_argument(
        "--monitor_interval", type=int, default=20, help="Steps between monitor model internals"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default="tokenizer_vocab",
        help="BPE tokenizer directory",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from (or 'auto' to auto-detect best checkpoint)",
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

    args = parser.parse_args()

    return args


# Wikitext-103 for pre-training Experimentations
DATAPATH = {
    "train": {
        "token_path": "Datasets/Pre_Training/dataset_train.bin",
    },
    "validation": {
        "token_path": "Datasets/Pre_Training/dataset_validation.bin",
    },
    "test": {
        "token_path": "Datasets/Pre_Training/dataset_test.bin",
    }
}

from Trainer.pretrainer import PreTrainer
if __name__ == "__main__":
    args = parse_arguments()
    initialize(args)
    
    match args.pipeline:
        case 'PT':
            print(f"\nStarted Training With {args.pipeline}")
            trainer = PreTrainer(args)
            total_time, best_val_loss = trainer.train(dataset= DATAPATH)
            
        case _:
            trainer = PreTrainer(args)
            print(f"\nStarted Training With Default pre-training Pipeline")
            total_time, best_val_loss = trainer.train(dataset= DATAPATH)

    print(f"\nTraining finished in {total_time:.2f} minutes.")
    print(f"Best Validation Loss achieved: {best_val_loss:.4f}")