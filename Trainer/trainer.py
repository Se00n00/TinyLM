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
    parser.add_argument(
        "--warmup_steps", type=int, default=2000, help="LR warmup steps"
    )
    parser.add_argument(
        "--eval_interval", type=int, default=200, help="Steps between evaluations"
    )
    parser.add_argument(
        "--monitor_interval",
        type=int,
        default=20,
        help="Steps between monitor model internals",
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

from datasets import load_dataset

# from Trainer.pretrainer import PreTrainer
from transformers import AutoTokenizer

from trainer import SFTConfig, SFTTrainer
from Model.layers import Config
from Model.models import Model
from datasets import load_dataset_builder

if __name__ == "__main__":
    args = parse_arguments()
    initialize(args)

    match args.pipeline:
        case "PT":
            print(f"\nStarted Training With {args.pipeline}")
            builder = load_dataset_builder("Se00n00/FineWeb-1B")
            total_samples = builder.info.splits["train"].num_examples


            tokenizer = AutoTokenizer.from_pretrained("Se00n00/TinyLM-2")
            match args.model:
                case "Alibi":
                    model = Model(Config(vocab_size=len(tokenizer)))
    
                case _:
                    model = Model(Config(vocab_size=len(tokenizer)))
                    
            trainer = SFTTrainer(
                training_name="Train",
                model=model,
                tokenizer= tokenizer,
                ds=load_dataset("Se00n00/FineWeb-1B", split="train", streaming=True),
                config = SFTConfig(total_samples=total_samples)
            )

            # trainer = PreTrainer(
            #     args, type="huggingface", data=PRETRAINING_HUGGINGFACE_DATASET
            # )
            # total_time, best_val_loss =
            trainer.train()

        case _:
            trainer = PreTrainer(args, type="local", data=LOCAL_DATAPATH)
            print("\nStarted Training With Default pre-training Pipeline")
            total_time, best_val_loss = trainer.train()

    print(f"\nTraining finished in {total_time:.2f} minutes.")
    print(f"Best Validation Loss achieved: {best_val_loss:.4f}")


# CHAT TEMPLATE 
# -------------------------
# For Text Genration
# -------------------------
# "text": ...
# -------------------------
# Instruction Fine Tunning
# -------------------------
# "messages":[
#     {
#         "role":"system",
#         "content":"..."
#     },
#     {
#         "role":"user",
#         "content":"..."
#     },
#     {
#         "role":"assistant",
#         "content": "..."
#     }
# ]
# -------------------------
# Reasoning Fine Tunning
# -------------------------
# "messages":[
#     {
#         "role":"system",
#         "content":"..."
#     },
#     {
#         "role":"user",
#         "content":"..."
#     },
#     {
#         "role":"assistant",
#         "content": "<|THINK|> ... <|THINK|> ..."
#     }
# ]
# -------------------------
# Tool Calling
# -------------------------
# "messages":[{
#     "available_tools": [
#         {
#             "name": "get_weather",
#             "description": "Get the current weather for a city.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "city": {
#                         "type": "string",
#                         "description": "Name of the city."
#                     }
#                 },
#                 "required": ["city"]
#             }
#         }
#     ],
#     "system": "...",
#     "user":"...",
#     "assisstant": """... <|TOOL_CALLS|>[
#         {
#             "name": "...",
#             "arguments":{
#                 "expression":"..."
#             }
#         }
#     ]<|/TOOL_CALLS|> ..."""
# }]
# 
# ---------------------------
# Tool Calling + Reasoning
# ---------------------------
# "messages":[
#     {
#         "role":"available_tools",
#         "content":[
#             {
#                 "name": "get_weather",
#                 "description": "Get the current weather for a city.",
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                         "city": {
#                             "type": "string",
#                             "description": "Name of the city."
#                         }
#                     },
#                     "required": ["city"]
#                 }
#             }
#         ]
#     },
#     {
#         "role": "system",
#         "content":"...",
#     },
#     {
#         "role": "user",
#         "content": "..."
#     },
#     {
#         "role":"assistant",
#         "content": """<|THINK|> ... <|/THINK|> ... <|TOOL_CALLS|>[
#             {
#                 "name": "...",
#                 "arguments":{
#                     "expression":"..."
#                 }
#             }
#         ]<|/TOOL_CALLS|> ..."""
#     }
# ]