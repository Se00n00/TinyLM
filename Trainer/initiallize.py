"""
Trainer Initiallizer initallizes the model checkpoints and logs
: checkpoint/<Training_Name>
    |- <Model_pipeline>.pt         # Checkpoints
    |- logs/
    |    |- train.csv
    |    |- validation.csv
    |- monitor/
         |- train.csv
         |- validation.csv
"""

import csv
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from Model.layers import Config
from Model.models import Model


def init_csv(csv_path, cols):
    if not Path(csv_path).exists():
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow(cols)


def initialize(args):
    os.makedirs(f"{args.checkpoint_dir}/{args.training_name}/{args.model}_{args.pipeline}", exist_ok=True)

    init_csv(
        f"{args.checkpoint_dir}/{args.training_name}/{args.model}_{args.pipeline}/logs_train.csv",
        ["step", "loss", "perplexity", "learning_rate", "GNorm"],
    )
    init_csv(
        f"{args.checkpoint_dir}/{args.training_name}/{args.model}_{args.pipeline}/logs_validation.csv",
        ["step", "loss", "perplexity", "learning_rate", "GNorm"],
    )
 
    init_csv(
        f"{args.checkpoint_dir}/{args.training_name}/{args.model}_{args.pipeline}/monitor_train.csv",
        ["step", "loss", "AttentionEntropy"],
    )
    init_csv(
        f"{args.checkpoint_dir}/{args.training_name}/{args.model}_{args.pipeline}/monitor_validation.csv",
        ["step", "loss", "AttentionEntropy"],
    )

    tokenizer = AutoTokenizer.from_pretrained("Se00n00/TinyLM-2")
    match args.model:
        case "Alibi":
            model = Model(
                Config(vocab_size=len(tokenizer))
            )
        case _:
            model = Model(
                Config(vocab_size=len(tokenizer))
            )

    checkpoint_path = Path(
        os.path.join(args.checkpoint_dir, f"{args.training_name}/{args.model}_{args.pipeline}.pt")
    )
    if checkpoint_path.is_file():
        print("The file exists.")
    else:
        torch.save({"model_state_dict": model.state_dict(), "step":-1}, checkpoint_path)

    print("\n----------------------------------------------------------------------------")
    print(f"\t\t INITIALLIZATION COMPLETE: {sum(p.numel() for p in model.parameters()) / (1024**2)}M")
    print("----------------------------------------------------------------------------")
    