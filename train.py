import argparse
import math
import os
import time
import numpy as np
import torch
from Datasets.tokenizer import BPETokenizer
from model import Config, Model
from Training import (
    check_and_cooldown_gpu,
    check_vram_limit,
    get_gpu_temperature,
    get_lr,
    prepare_datasets,
)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Tiny Parrot")

    MODEL_HELP = """
    |--------------------------------------|
        AVAILABLE MODELS
    |--------------------------------------|
        GPT : GPT-2 Style Transformer
    """

    parser.add_argument(
        "--model", type=str, choices=["GPT"], default="GPT", help=MODEL_HELP
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
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
        "--vram_limit_mb",
        type=int,
        default=3500,
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


def get_batch(step, data, data_len, batch_size, max_seq_len, device):
    """
    Fetches a sequential batch of X and Y from tokenized binary data.
    """
    start = (step * batch_size * max_seq_len) % (data_len - max_seq_len - 1)

    ix = torch.arange(start, start + batch_size * max_seq_len, max_seq_len)

    x = torch.stack(
        [torch.from_numpy(data[i : i + max_seq_len].astype(np.int64)) for i in ix]
    )

    y = torch.stack(
        [torch.from_numpy(data[i + 1 : i + 1 + max_seq_len].astype(np.int64)) for i in ix]
    )

    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate_loss(model, data, batch_size, max_seq_len, device):
    model.eval()

    losses = []

    max_steps = (len(data) - 1) // (batch_size * max_seq_len)

    for step in range(max_steps):
        x, y = get_batch(step, data, len(data), batch_size, max_seq_len, device)

        _, loss = model(x, y)
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


DATAPATH = {
    "train": {
        "token_path": "Datasets/Tokens/train.bin",
        "dataset_path": "Datasets/train",
    },
    "validation": {
        "token_path": "Datasets/Tokens/val.bin",
        "dataset_path": "Datasets/validation",
    },
    "test": {"token_path": "Datasets/Tokens/test.bin", "dataset_path": "Datasets/test"},
}

if __name__ == "__main__":
    args = parse_arguments()

    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. ENCODE DATASET
    tokenizer = BPETokenizer(path="Datasets/tokenizer.json")
    prepare_datasets(DATAPATH)

    # 2. LOAD MEMMAPPED TOKEN DATA
    train_data = np.memmap("Datasets/Tokens/train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap("Datasets/Tokens/val.bin", dtype=np.uint16, mode="r")

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
        case _:
            model = Model(Config(vocab_size=tokenizer.vocab_size, block_size=args.max_seq_len))

    # if args.gradient_checkpointing:
    #     model.gradient_checkpointing = True
    #     print("Gradient Checkpointing enabled from scratch.")

    model.to(device)

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
            resume_path = os.path.join(args.checkpoint_dir, f"best_{args.model}.pt")

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

    t0 = time.time()

    # Save a run metadata JSON
    run_meta = {
        "model": args.model,
        "vocab_size": tokenizer.vocab_size,
        "batch_size": args.batch_size,
        "max_seq_len": args.max_seq_len,
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
    }

    # Training state
    model.train()

    # Track micro-batch parameters dynamically
    BATCH_SIZE = args.batch_size  # 8
    GRAD_ACCUM_STEPS = args.grad_accum_steps  # 4
    EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRAD_ACCUM_STEPS  # 32
    
    datalen = len(train_data)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(
        "\n--------------------------------------------------------------------------"
    )
    print(f"DATASET: TRAIN TOKENS {datalen:,}| VALIDATION TOKENS: {len(val_data):,}")
    print(f"DEVICE: {device} MODEL: {args.model}| PARAMETERS:{num_params/(1024*1024)}| BATCH SIZE:{BATCH_SIZE}")
    print(
        "----------------------------------------------------------------------------\n"
    )

    while step < args.max_steps:
        # Check GPU temperature and pause if needed
        check_and_cooldown_gpu(args.max_temp, args.cooldown_temp)

        # Calculate learning rate
        lr = get_lr(step, args.max_steps, args.learning_rate, args.warmup_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        t_start = time.time()
        step_completed = False

        # Self-healing OOM/limit recovery loop
        while not step_completed:
            try:
                optimizer.zero_grad(set_to_none=True)
                loss_accum = 0.0

                # Check current allocated VRAM
                check_vram_limit(args.vram_limit_mb, device)

                # Use autocast context
                from torch.amp import autocast

                device_type = "cuda" if device.type == "cuda" else "cpu"

                for micro_step in range(GRAD_ACCUM_STEPS):
                    global_micro_step = step*GRAD_ACCUM_STEPS + micro_step

                    x, y = get_batch(
                        global_micro_step, train_data, datalen, BATCH_SIZE, args.max_seq_len, device
                    )

                    with autocast(device_type=device_type, dtype=ptdtype):
                        logits, loss = model(x, y)
                        loss = loss / GRAD_ACCUM_STEPS

                    loss_accum += loss.item()

                    if ptdtype == torch.float16:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

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
        if step % args.eval_interval == 0 or step == args.max_steps - 1:
            val_loss = evaluate_loss(
                model,
                val_data,
                BATCH_SIZE,
                args.max_seq_len,
                device
            )
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

            print(
                f"Step {step:4d}/{args.max_steps:4d} | "
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
                    args.checkpoint_dir, f"best_{args.model}.pt"
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
                f"Step {step:4d}/{args.max_steps:4d} | Train Loss: {loss_accum:.4f} | LR: {lr:.2e} | VRAM: {allocated_mb:.1f}MB{temp_str} | Time: {step_time_ms:.1f}ms",
                end="\r",
            )

        step += 1

    total_time_min = (time.time() - t0) / 60
    print(f"\nTraining finished in {total_time_min:.2f} minutes.")
    print(f"Best Validation Loss achieved: {best_val_loss:.4f}")
