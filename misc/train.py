import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import math
import os
import time
import numpy as np
import torch
from Datasets.tokenizer import BPETokenizer
from model import Config, Model
from Trainer import (
    check_and_cooldown_gpu,
    check_vram_limit,
    get_gpu_temperature,
    get_lr,
    prepare_datasets,
)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Tiny Parrot")

    parser.add_argument(
        "--training_name", type=str
    )
    parser.add_argument(
        "--model", type=str, choices=["GPT"], default="GPT"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Micro-batch size (reduced to fit in VRAM)",
    )
    parser.add_argument(
        "--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=512, help="Maxiumum Sequence length"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="Max learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.1, help="Weight Decay rate"
    )

    parser.add_argument(
        "--max_steps", type=int, default=20000, help="Total training steps"
    )
    parser.add_argument("--warmup_steps", type=int, default=200, help="LR warmup steps")
    parser.add_argument(
        "--eval_interval", type=int, default=200, help="Steps between evaluations"
    )
    parser.add_argument(
        "--eval_iters", type=int, default=50, help="Evaluation iterations"
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
        choices=["PT","IFT","PFT"], # Pre-training, Instruction Finetunning, Preference Fine-Tunning
        default='PT',
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


def get_batch(
    step,
    data,
    data_len,
    batch_size,
    max_seq_len,
    device,
    pipeline,
    rand=False,
):
    """
    Fetches a batch for Pretraining / IFT / PFT.
    """

    batch = {}

    match pipeline:

        # -------------------------------------------------
        # IFT
        # -------------------------------------------------
        case "IFT":

            start = (step * batch_size * max_seq_len) % (
                data_len - max_seq_len - 1
            )

            if rand:
                ix = torch.randint(
                    start,
                    data_len - max_seq_len - 1,
                    (batch_size,),
                )
            else:
                ix = torch.arange(
                    start,
                    start + batch_size * max_seq_len,
                    max_seq_len,
                )

            batch["inputs"] = torch.stack([
                torch.from_numpy(
                    data["token_path"][i:i + max_seq_len].astype(np.int64)
                )
                for i in ix
            ]).to(device)

            batch["labels"] = torch.stack([
                torch.from_numpy(
                    data["label_path"][i:i + max_seq_len].astype(np.int64)
                )
                for i in ix
            ]).to(device)

        # -------------------------------------------------
        # PFT / DPO
        # -------------------------------------------------
        case "PFT":
            chosen_offsets = data["chosen_offset_path"]
            rejected_offsets = data["rejected_offset_path"]
            
            assert len(chosen_offsets) == len(rejected_offsets)
            
            num_examples = len(chosen_offsets) - 1

            if rand:
                ix = torch.randint(
                    0,
                    num_examples,
                    (batch_size,),
                )
            else:
                start = (step * batch_size) % num_examples
                end = min(start + batch_size, num_examples)
                ix = torch.arange(start, end)

            chosen_inputs = []
            chosen_labels = []

            rejected_inputs = []
            rejected_labels = []

            chosen_offsets = data["chosen_offset_path"]
            rejected_offsets = data["rejected_offset_path"]

            for idx in ix.tolist():
            
                # --------------------
                # chosen
                # --------------------
                s = chosen_offsets[idx]
                e = chosen_offsets[idx + 1]
            
                chosen_ids = data["chosen_token_path"][s:e][:max_seq_len]
                chosen_lbl = data["chosen_label_path"][s:e][:max_seq_len]
            
                chosen_inputs.append(
                    torch.from_numpy(chosen_ids.astype(np.int64))
                )
            
                chosen_labels.append(
                    torch.from_numpy(chosen_lbl.astype(np.int64))
                )
            
                # --------------------
                # rejected
                # --------------------
                s = rejected_offsets[idx]
                e = rejected_offsets[idx + 1]
            
                rejected_ids = data["rejected_token_path"][s:e][:max_seq_len]
                rejected_lbl = data["rejected_label_path"][s:e][:max_seq_len]
            
                rejected_inputs.append(
                    torch.from_numpy(rejected_ids.astype(np.int64))
                )
            
                rejected_labels.append(
                    torch.from_numpy(rejected_lbl.astype(np.int64))
                )
                
            batch["chosen_input_ids"] = torch.nn.utils.rnn.pad_sequence(
                chosen_inputs,
                batch_first=True,
                padding_value=0,
            ).to(device)

            batch["chosen_labels"] = torch.nn.utils.rnn.pad_sequence(
                chosen_labels,
                batch_first=True,
                padding_value=0,
            ).to(device)

            batch["rejected_input_ids"] = torch.nn.utils.rnn.pad_sequence(
                rejected_inputs,
                batch_first=True,
                padding_value=0,
            ).to(device)

            batch["rejected_labels"] = torch.nn.utils.rnn.pad_sequence(
                rejected_labels,
                batch_first=True,
                padding_value=0,
            ).to(device)

            batch["chosen_attention_mask"] = (
                batch["chosen_input_ids"] != 0
            ).long().to(device)

            batch["rejected_attention_mask"] = (
                batch["rejected_input_ids"] != 0
            ).long().to(device)

        # -------------------------------------------------
        # Pretraining
        # -------------------------------------------------
        case _:

            start = (step * batch_size * max_seq_len) % (
                data_len - max_seq_len - 1
            )

            if rand:
                ix = torch.randint(
                    start,
                    data_len - max_seq_len - 1,
                    (batch_size,),
                )
            else:
                ix = torch.arange(
                    start,
                    start + batch_size * max_seq_len,
                    max_seq_len,
                )

            batch["inputs"] = torch.stack([
                torch.from_numpy(
                    data["token_path"][i:i + max_seq_len].astype(np.int64)
                )
                for i in ix
            ]).to(device)

            batch["labels"] = torch.stack([
                torch.from_numpy(
                    data["token_path"][i + 1:i + 1 + max_seq_len].astype(np.int64)
                )
                for i in ix
            ]).to(device)

    return batch

from tqdm import tqdm

@torch.no_grad()
def evaluate_loss(model, reference_model, reference_checkpoint_dir, data, data_len, batch_size, max_seq_len, device, pipeline:str, eval_steps=1000):
    model.eval()

    total_loss = 0.0

    skipped_batch = 0
    for step in tqdm(range(eval_steps)):
        batch = get_batch(
            step,
            data,
            data_len,
            batch_size,
            max_seq_len,
            device,
            pipeline,
            
            rand = True
        )
        if (
            batch["chosen_input_ids"].shape[1] == 0
            or batch["chosen_labels"].shape[1] == 0
            or batch["rejected_input_ids"].shape[1] == 0
            or batch["rejected_labels"].shape[1] == 0
        ):
            # print("Skipping empty sequence batch.")
            skipped_batch += 1
            continue
            
        if pipeline == 'PFT':
            loss, chosen_reward, rejected_reward = dpo_loss(
                policy_model = model,
                reference_model_checkpoint_dir = reference_checkpoint_dir,
                reference_model_name = reference_model,
                chosen_input_ids = batch["chosen_input_ids"],
                chosen_labels = batch["chosen_labels"],
                rejected_input_ids = batch["rejected_input_ids"],
                rejected_labels = batch["rejected_labels"],
                beta = 0.1,
                device_type = device_type,
                dtype = ptdtype
            )
        else:
            with autocast(device_type=device_type, dtype=ptdtype):
                logits, loss = model(batch['inputs'], batch['labels'])
        
        if torch.isnan(loss):
            skipped_batch += 1
            continue
        total_loss += loss.item()
        
    torch.cuda.empty_cache()

    model.train()

    return total_loss / (eval_steps -skipped_batch)

import csv
from pathlib import Path
from dpo_trainer import dpo_loss

def init_csv(csv_path):
    if not Path(csv_path).exists():
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                "step",
                "loss",
                "perplexity",
                "learning_rate"
            ])
def log_metrics(csv_path, step, loss, lr):
    file = open(csv_path, "a", newline="")
    
    writer = csv.writer(file)
    writer.writerow([
        step,
        loss,
        math.exp(loss),
        lr
    ])
    
    file.flush()
        
# DATAPATH = {
#     "train": {
#         "token_path": "Datasets/Tokens/sft_train_tokens.bin",
#         "label_path": "Datasets/Tokens/sft_train_labels.bin"
#     },
#     "validation": {
#         "token_path": "Datasets/Tokens/sft_val_tokens.bin",
#         "label_path": "Datasets/Tokens/sft_val_labels.bin"
#     }
# }
DATAPATH = {
    "train": {
        "chosen_token_path": "Datasets/Tokens/pft_chosen_tokens.bin",
        "chosen_label_path": "Datasets/Tokens/pft_chosen_labels.bin",
        "rejected_token_path": "Datasets/Tokens/pft_rejected_tokens.bin",
        "rejected_label_path": "Datasets/Tokens/pft_rejected_labels.bin",
        "chosen_offset_path":"Datasets/Tokens/pft_chosen_offsets.bin",
        "rejected_offset_path":"Datasets/Tokens/pft_rejected_offsets.bin",
    },
    "validation": {
        "chosen_token_path": "Datasets/Tokens/pft_chosen_tokens.bin",
        "chosen_label_path": "Datasets/Tokens/pft_chosen_labels.bin",
        "rejected_token_path": "Datasets/Tokens/pft_rejected_tokens.bin",
        "rejected_label_path": "Datasets/Tokens/pft_rejected_labels.bin",
        "chosen_offset_path":"Datasets/Tokens/pft_chosen_offsets.bin",
        "rejected_offset_path":"Datasets/Tokens/pft_rejected_offsets.bin",
    }
}

from typing import Dict, Any
def load_dataset(datapath: Dict[str, Any]):
    data = {}

    for split, files in datapath.items():
        data[split] = {}

        for name, path in files.items():
            data[split][name] = np.memmap(
                path,
                dtype=np.uint16,
                mode="r"
            )

    return data

import numpy as np
import torch
import torch.nn.functional as F


if __name__ == "__main__":
    args = parse_arguments()
    init_csv(f"logs/train_{args.training_name}.csv")
    init_csv(f"logs/validation_{args.training_name}.csv")
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. ENCODE DATASET
    tokenizer = BPETokenizer(path="Datasets/tokenizer.json")
    # prepare_datasets(DATAPATH)

    # 2. LOAD MEMMAPPED TOKEN DATA
    Data = load_dataset(DATAPATH)
    
    # train_data = np.memmap(DATAPATH['train']['token_path'], dtype=np.uint16, mode="r")
    # val_data = np.memmap(DATAPATH['validation']['token_path'], dtype=np.uint16, mode="r")
    
    # if args.pipeline == 'IFT':
    #     train_data_labels = np.memmap(DATAPATH['train']['label_path'], dtype=np.uint16, mode="r")
    #     val_data_labels = np.memmap(DATAPATH['validation']['label_path'], dtype=np.uint16, mode="r")

    # 4. Instantiate Model and Setup Device / Precision
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        # Check and enforce the VRAM limit fraction
        curr_dev = torch.cuda.current_device()
        total_mem = torch.cuda.get_device_properties(curr_dev).total_memory / (
            1024 * 1024
        )
        if args.vram_limit_mb < total_mem:
            fraction = args.vram_limit_mb / total_mem
            torch.cuda.set_per_process_memory_fraction(fraction, device=curr_dev)
            print(
                f"[Memory Guard] Configured CUDA memory fraction to {fraction:.4f} (limits VRAM to ~{args.vram_limit_mb} MB of total {total_mem:.1f} MB)"
            )
        else:
            print(
                f"[Memory Guard] Requested VRAM limit ({args.vram_limit_mb} MB) exceeds total device VRAM ({total_mem:.1f} MB). Limit not set."
            )

        # Determine AMP data type
        if not args.disable_amp:
            ptdtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
            print(f"Automatic Mixed Precision (AMP) enabled with: {ptdtype}")
        else:
            ptdtype = torch.float16
            print("AMP disabled. Using standard FP16 precision.")
    else:
        ptdtype = torch.float16
        print("Using standard FP16 precision on CPU.")

    # Setup GradScaler for float16
    scaler = torch.amp.GradScaler("cuda", enabled=(ptdtype == torch.float16))

    match args.model:
        case "gpt2":
            model = Model(Config(vocab_size=tokenizer.vocab_size, block_size=args.max_seq_len))
            # if args.pipeline == 'PFT':
            #     reference_model = Model(Config(vocab_size=tokenizer.vocab_size, block_size=args.max_seq_len))
            
        case _:
            model = Model(Config(vocab_size=tokenizer.vocab_size, block_size=args.max_seq_len))
            # if args.pipeline == 'PFT':
            #     reference_model = Model(Config(vocab_size=tokenizer.vocab_size, block_size=args.max_seq_len))
            
    # if args.gradient_checkpointing:
    #     model.gradient_checkpointing = True
    #     print("Gradient Checkpointing enabled from scratch.")
    
    model.to(device)
    # if args.pipeline == 'PFT':
    #     reference_model.to(device)
    
    
    args.learning_rate = args.learning_rate if args.pipeline == "PT" else 1e-4
    # 5. Setup Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), weight_decay=args.weight_decay, lr=args.learning_rate
    )

    # 6. Training loop
    step = 0
    best_val_loss = float("inf")

    # Optional checkpoint resumption
    if args.resume is not None:
        resume_path = args.resume
        if resume_path.lower() == "auto":
            resume_path = os.path.join(args.checkpoint_dir, f"{args.model}_{args.pipeline}.pt")

        if os.path.exists(resume_path):
            print(f"Resuming training from checkpoint: {resume_path}")
            checkpoint = torch.load(
                resume_path, map_location=device, weights_only=False
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Move optimizer states to target device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

            step = checkpoint["step"] + 1
            best_val_loss = checkpoint.get("val_loss", float("inf"))
            print(
                f"Resumed successfully. Continuing from step {step} with best val loss: {best_val_loss:.4f}"
            )
        else:
            print(
                f"Warning: Checkpoint path '{resume_path}' not found. Starting training from scratch (step 0)."
            )
            
    if args.pipeline == 'IFT' and args.resume is None:
        resume_path = os.path.join(args.checkpoint_dir, f"{args.model}_{args.pipeline}.pt")
        if os.path.exists(resume_path):
            print(f"Resuming training from checkpoint: {resume_path}")
            checkpoint = torch.load(
                resume_path, map_location=device, weights_only=False
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            print(
                f"Resumed successfully. Continuing from step {step} with best val loss: {best_val_loss:.4f}"
            )
        else:
            print(
                f"Warning: Checkpoint path '{resume_path}' not found. Starting training from scratch (step 0)."
            )
    
    if args.pipeline == 'PFT' and args.resume is None:
        resume_path = os.path.join(args.checkpoint_dir, f"{args.model}_{args.pipeline}.pt")
        reference_resume_path = os.path.join(args.checkpoint_dir, f"{args.model}_IFT.pt")
        
        if os.path.exists(resume_path):
            print(f"Resuming training from checkpoint: {resume_path}")
            checkpoint = torch.load(
                resume_path, map_location=device, weights_only=False
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            print(
                f"Resumed successfully. Continuing from step {step} with best val loss: {best_val_loss:.4f}"
            )
        else:
            print(
                f"Warning: Checkpoint path '{resume_path}' not found. Starting training from scratch (step 0)."
            )
        
        # if os.path.exists(reference_resume_path):
        #     print(f"Loading Reference Model: {reference_resume_path}")
        #     reference_checkpoint = torch.load(
        #         reference_resume_path, map_location=device, weights_only=False
        #     )
        #     reference_model.load_state_dict(reference_checkpoint["model_state_dict"])
        #     print(
        #         f"Resumed successfully. Continuing from step {step} with best val loss: {best_val_loss:.4f}"
        #     )
        #     reference_model.eval()
            
        #     for p in reference_model.parameters():
        #         p.requires_grad = False
        # else:
        #     print(
        #         f"Warning: Checkpoint path '{resume_path}' not found. Starting training from scratch (step 0)."
        #     )
    t0 = time.time()

    

    # Training state
    model.train()

    # Track micro-batch parameters dynamically
    BATCH_SIZE = args.batch_size  # 8
    GRAD_ACCUM_STEPS = args.grad_accum_steps  # 4
    EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRAD_ACCUM_STEPS  # 32
    
    if args.pipeline == 'PFT':
        train_data = Data['train']['chosen_token_path']
        val_data = Data['validation']['chosen_token_path']
        
    else:
        train_data = Data['train']['token_path']
        val_data = Data['validation']['token_path']
        
    train_len = len(train_data)
    val_len = len(val_data)
    
    num_params = sum(p.numel() for p in model.parameters())
    
    tokens_per_step = (
        BATCH_SIZE
        * args.max_seq_len
        * GRAD_ACCUM_STEPS
    )

    steps_per_epoch = math.ceil(
        train_len / tokens_per_step
    )
    # Save a run metadata JSON
    run_meta = {
        "model": args.model,
        "vocab_size": tokenizer.vocab_size,
        "batch_size": args.batch_size,
        "max_seq_len": args.max_seq_len,
        "learning_rate": args.learning_rate,
        "max_steps": steps_per_epoch,
    }
    print(
        "\n--------------------------------------------------------------------------"
    )
    print(f"DATASET: TRAIN TOKENS {train_len:,}| VALIDATION TOKENS: {val_len:,} | TOKENS PER STEPS: {tokens_per_step}| STEPS PER EPOCH: {steps_per_epoch}")
    print(f"DEVICE: {device} MODEL: {args.model}| PARAMETERS:{num_params/(1024*1024)}| BATCH SIZE:{BATCH_SIZE}")
    print(
        "----------------------------------------------------------------------------\n"
    )


    while step < steps_per_epoch:
        
        # Check GPU temperature and pause if needed
        check_and_cooldown_gpu(args.max_temp, args.cooldown_temp)

        # Calculate learning rate
        lr = get_lr(step, steps_per_epoch, args.learning_rate, args.warmup_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        t_start = time.time()
        step_completed = False

        # Self-healing OOM/limit recovery loop
        while not step_completed:
            torch.cuda.empty_cache()
            try:
                #print("INITIAL ",torch.cuda.memory_allocated()/1024**2,"\n")
                optimizer.zero_grad(set_to_none=True)
                loss_accum = 0.0

                # Check current allocated VRAM
                check_vram_limit(args.vram_limit_mb, device)

                # Use autocast context
                from torch.amp import autocast

                device_type = "cuda" if device.type == "cuda" else "cpu"
                
                skipped_batch = 0
                for micro_step in range(GRAD_ACCUM_STEPS):
                    
                    global_micro_step = step*GRAD_ACCUM_STEPS + micro_step

                    # x, y = get_batch(
                    #     global_micro_step, train_data, datalen, BATCH_SIZE, args.max_seq_len, device, instruction_set = True, labels = train_data_labels
                    # )
                    # print(y.dtype)
                    # print(y.min().item(), y.max().item())
                    # print(tokenizer.vocab_size)
                    #print("AFTER TRAIN DATASET ",torch.cuda.memory_allocated()/1024**2,"\n")

                    # with autocast(device_type=device_type, dtype=ptdtype):
                    #     logits, loss = model(x, y)
                        
                        
                    #print("AFTER TRAIN FORWARD ",torch.cuda.memory_allocated()/1024**2,"\n")
                    # ---------------------------------------------------------------------
                    batch = get_batch(
                        global_micro_step, Data['train'], train_len, BATCH_SIZE, args.max_seq_len, device, args.pipeline
                    )
                    
                    if (
                        batch["chosen_input_ids"].shape[1] == 0
                        or batch["chosen_labels"].shape[1] == 0
                        or batch["rejected_input_ids"].shape[1] == 0
                        or batch["rejected_labels"].shape[1] == 0
                    ):
                        print("Skipping empty sequence batch.")
                        skipped_batch += 1
                        continue
                    
                    if args.pipeline == 'PFT':
                        loss, chosen_reward, rejected_reward = dpo_loss(
                            policy_model = model,
                            reference_model_checkpoint_dir = args.checkpoint_dir,
                            reference_model_name = args.model,
                            chosen_input_ids = batch["chosen_input_ids"],
                            chosen_labels = batch["chosen_labels"],
                            rejected_input_ids = batch["rejected_input_ids"],
                            rejected_labels = batch["rejected_labels"],
                            beta = 0.1,
                            device_type = device_type,
                            dtype = ptdtype
                        )
                    else:
                        
                        with autocast(device_type=device_type, dtype=ptdtype):
                            logits, loss = model(batch['inputs'], batch['labels']) # X, Y
                    
                    loss = loss / (GRAD_ACCUM_STEPS-skipped_batch)
                    
                    if torch.isnan(loss):
                        print("NaN loss encountered, skipping batch")
                        optimizer.zero_grad(set_to_none=True)
                        continue

                    loss_accum += loss.item()

                    if ptdtype == torch.float16:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                        
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    # 
                        
                    #print("AFTER TRAIN BACKWARD ",torch.cuda.memory_allocated()/1024**2,"\n")


                # Step the optimizer
                if ptdtype == torch.float16:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                step_completed = True

            except RuntimeError as e:
                err_msg = str(e).lower()
                if (
                    "out of memory" in err_msg
                    or "memory limit" in err_msg
                    or "allowed memory" in err_msg
                ):
                    print(
                        f"\n[Memory Guard] CUDA OOM or limit exceeded at step {step} with batch_size={BATCH_SIZE}, grad_accum_steps={GRAD_ACCUM_STEPS}."
                    )
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()

                    if BATCH_SIZE > 1:
                        old_bs = BATCH_SIZE
                        BATCH_SIZE = max(1, BATCH_SIZE // 2)
                        GRAD_ACCUM_STEPS = EFFECTIVE_BATCH_SIZE // BATCH_SIZE
                        print(
                            f"  [Memory Guard] Halving micro-batch size: {old_bs} -> {BATCH_SIZE}. Increasing grad_accum_steps to {GRAD_ACCUM_STEPS}."
                        )
                    elif not getattr(model, "gradient_checkpointing", False):
                        print(
                            "  [Memory Guard] Micro-batch size is already 1. Enabling gradient checkpointing to save memory..."
                        )
                        model.gradient_checkpointing = True
                        # Reset batch size to original/default to try to recover with checkpointing
                        BATCH_SIZE = args.batch_size
                        GRAD_ACCUM_STEPS = args.grad_accum_steps
                    else:
                        print(
                            "  [Memory Guard] Out of memory even with micro-batch size 1 and gradient checkpointing active."
                        )
                        raise e
                else:
                    raise e

        t_end = time.time()
        step_time_ms = (t_end - t_start) * 1000

        # Periodic evaluation & logging
        if step % args.eval_interval == 0 or step == steps_per_epoch - 1:
            #print("BEFORE VAL",torch.cuda.memory_allocated()/1024**2,"\n")

            val_loss = evaluate_loss(
                model,
                args.model,
                args.checkpoint_dir,
                Data['validation'],
                val_len,
                1,
                args.max_seq_len,
                device,
                args.pipeline
            )
            #print("AFTER VAL",torch.cuda.memory_allocated()/1024**2,"\n")

            val_ppl = math.exp(val_loss) if val_loss < 30 else float("inf")
            train_ppl = math.exp(loss_accum) if loss_accum < 30 else float("inf")

            allocated_mb = (
                torch.cuda.memory_allocated(device) / (1024 * 1024)
                if device.type == "cuda"
                else 0.0
            )
            temp_str = (
                f" | Temp: {get_gpu_temperature()}°C"
                if get_gpu_temperature() is not None
                else ""
            )
            log_metrics(f"logs/validation_{args.training_name}.csv", step, val_loss, 0)

            print(
                f"Step {step:4d}/{steps_per_epoch:4d} | "
                f"Train Loss: {loss_accum:.4f} (PPL: {train_ppl:.2f}) | "
                f"Val Loss: {val_loss:.4f} (PPL: {val_ppl:.2f}) | "
                f"LR: {lr:.2e} | "
                f"Grad Norm: {grad_norm:.2f} | "
                f"Step Time: {step_time_ms:.1f}ms | "
                f"VRAM: {allocated_mb:.1f}MB" + temp_str
            )

            # Save checkpoint if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(
                    args.checkpoint_dir, f"{args.model}_{args.pipeline}.pt"
                )
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "args": args,
                        "run_meta": run_meta,
                    },
                    checkpoint_path,
                )
                print(
                    f"  [Checkpoint] Saved best {args.model} model to {checkpoint_path} (Val Loss: {val_loss:.4f})"
                )

        # Basic print on progress
        elif step % 20 == 0:
            allocated_mb = (
                torch.cuda.memory_allocated(device) / (1024 * 1024)
                if device.type == "cuda"
                else 0.0
            )
            temp_str = (
                f" | Temp: {get_gpu_temperature()}°C"
                if get_gpu_temperature() is not None
                else ""
            )
            print(
                f"Step {step:4d}/{steps_per_epoch:4d} | Train Loss: {loss_accum:.4f} | LR: {lr:.2e} | VRAM: {allocated_mb:.1f}MB{temp_str} | Time: {step_time_ms:.1f}ms",
                end="\r",
            )
        
        log_metrics(f"logs/train_{args.training_name}.csv", step, loss_accum, lr)
        step += 1

    total_time_min = (time.time() - t0) / 60
    print(f"\nTraining finished in {total_time_min:.2f} minutes.")
    print(f"Best Validation Loss achieved: {best_val_loss:.4f}")
