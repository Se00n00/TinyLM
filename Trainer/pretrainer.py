import logging
import math
import os
import time
from typing import Any, Dict

import numpy as np
import torch
from torch.amp import autocast
from tqdm import tqdm
from transformers import AutoTokenizer

from Model.layers import Config
from Model.loss import get_cross_entropy_loss
from Model.models import Model
from Trainer.base import Trainer
from Trainer.util import get_lr

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

class PreTrainer(Trainer):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

    def train(
        self, hugginface_dataset: None | Dict = None, dataset: None | Dict = None
    ):
        super().train()

        # 1. Load / Setup Dataset
        if dataset:
            Data = self.load_dataset_from_local(dataset)
        elif hugginface_dataset:
            pass

            # Else setup Streaming for Pre-training using hugginface dataset

        # 2. Instantiate Model, Tokenizer, Optimizer and Setup Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if device.type == "cuda":
            self.enforce_gpu_limit(self.args.vram_limit_mb)

        ptdtype = torch.float16
        scaler = torch.amp.GradScaler(device.type, enabled=(ptdtype == torch.float16))

        tokenizer = AutoTokenizer.from_pretrained("Se00n00/TinyLM-2")
        match self.args.model:
            case "Alibi":
                model = Model(Config(vocab_size=len(tokenizer)))

            case _:
                model = Model(Config(vocab_size=len(tokenizer)))

        num_params = sum(p.numel() for p in model.parameters())
        model.to(device)

        self.args.learning_rate = (
            self.args.learning_rate
            if self.args.learning_rate != 0.0
            else self.get_learning_rate(self.args.pipeline)
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            weight_decay=self.args.weight_decay,
            lr=self.args.learning_rate,
        )

        step = 0
        best_val_loss = float("inf")
        val_loss = float("inf")
        # Optional checkpoint resumption
        if self.args.resume:
            resume_path = self.args.resume
            if resume_path.lower() == "auto":
                resume_path = os.path.join(
                    self.args.checkpoint_dir,
                    f"{self.args.training_name}/{self.args.model}_{self.args.pipeline}.pt",
                )

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

                step = checkpoint.get("step", 0) + 1
                best_val_loss = checkpoint.get("val_loss", float("inf"))
                print(
                    f"Resumed successfully. Continuing from step {step} with best val loss: {best_val_loss:.4f}"
                )
            else:
                print(
                    f"Warning: Checkpoint path '{resume_path}' not found. Starting training from scratch (step 0)."
                )

        model.train()

        if dataset:
            train_data = Data["train"]["token_path"]
            val_data = Data["validation"]["token_path"]
            test_data = Data["test"]["token_path"]

            train_len = len(train_data)
            val_len = len(val_data)
            test_len = len(test_data)

        BATCH_SIZE = self.args.batch_size  # 8
        GRAD_ACCUM_STEPS = self.args.grad_accum_steps  # 4
        EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRAD_ACCUM_STEPS  # 32
        tokens_per_step = BATCH_SIZE * self.args.max_seq_len * GRAD_ACCUM_STEPS

        steps_per_epoch = math.ceil(train_len / tokens_per_step)
        # Save a run metadata JSON
        run_meta = {
            "model": self.args.model,
            "vocab_size": len(tokenizer),
            "batch_size": self.args.batch_size,
            "max_seq_len": self.args.max_seq_len,
            "learning_rate": self.args.learning_rate,
            "max_steps": steps_per_epoch,
        }
        print(
            "\n--------------------------------------------------------------------------"
        )
        print(
            f"TOKENS/PARAMS:{num_params / train_len} : TRAIN TOKENS {train_len}| VALIDATION TOKENS: {val_len} | TOKENS/STEPS: {tokens_per_step}| STEPS/EPOCH: {steps_per_epoch}"
        )
        print(
            f"DEVICE: {device.type} MODEL: {self.args.model}| PARAMETERS:{num_params / (1024 * 1024)}| BATCH SIZE:{BATCH_SIZE}"
        )
        print(
            "----------------------------------------------------------------------------\n"
        )

        t0 = time.time()
        pbar1 = tqdm(total=steps_per_epoch, desc="Train")
        while step < steps_per_epoch:
            self.check_and_cooldown_gpu(self.args.max_temp, self.args.cooldown_temp)
            lr = get_lr(
                step, steps_per_epoch, self.args.learning_rate, self.args.warmup_steps
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            t_start = time.time()
            step_completed = False

            while not step_completed:
                torch.cuda.empty_cache()

                try:
                    optimizer.zero_grad(set_to_none=True)
                    loss_accum = 0.0

                    self.check_vram_limit(self.args.vram_limit_mb, device)
                    skipped_batch = 0
                    for micro_step in range(GRAD_ACCUM_STEPS):
                        global_micro_step = step * GRAD_ACCUM_STEPS + micro_step

                        batch = self.get_batch(
                            global_micro_step,
                            Data["train"],
                            train_len,
                            BATCH_SIZE,
                            self.args.max_seq_len,
                            device,
                            self.args.pipeline,
                        )

                        with autocast(device_type=device.type, dtype=ptdtype):
                            logits = model(batch["inputs"])
                            loss = get_cross_entropy_loss(logits, batch["labels"])

                        loss = loss / (GRAD_ACCUM_STEPS - skipped_batch)

                        if torch.isnan(loss):
                            print("NaN loss encountered, skipping batch")
                            optimizer.zero_grad(set_to_none=True)
                            continue

                        loss_accum += loss.item()

                        if ptdtype == torch.float16:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()

                    if ptdtype == torch.float16:
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 1.0
                        )
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 1.0
                        )
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
                            BATCH_SIZE = self.args.batch_size
                            GRAD_ACCUM_STEPS = self.args.grad_accum_steps
                        else:
                            print(
                                "  [Memory Guard] Out of memory even with micro-batch size 1 and gradient checkpointing active."
                            )
                            raise e
                    else:
                        raise e

            pbar1.update(1)
            t_end = time.time()
            step_time_ms = (t_end - t_start) * 1000

            # Evaluate
            if step % self.args.eval_interval == 0 or step == steps_per_epoch - 1:
                val_loss = self.evaluate_loss(
                    model,
                    self.args.model,
                    self.args.checkpoint_dir,
                    Data["validation"],
                    val_len,
                    1,
                    self.args.max_seq_len,
                    device,
                    self.args.pipeline,
                )
                val_ppl = math.exp(val_loss) if val_loss < 1000 else float("inf")
                train_ppl = math.exp(loss_accum) if loss_accum < 1000 else float("inf")
                
                allocated_mb = (
                    torch.cuda.memory_allocated(device) / (1024 * 1024)
                    if device.type == "cuda"
                    else 0.0
                )
                temp_str = (
                    f" | Temp: {self.get_gpu_temperature()}°C"
                    if self.get_gpu_temperature() is not None
                    else ""
                )
                # logger.info(
                #     f"Step {step:4d}/{steps_per_epoch:4d} | "
                #     f"Train Loss: {loss_accum:.4f} (PPL: {train_ppl:.2f}) | "
                #     f"Val Loss: {val_loss:.4f} (PPL: {val_ppl:.2f}) | "
                #     f"LR: {lr:.2e} | "
                #     f"Grad Norm: {grad_norm:.2f} | "
                #     f"Step Time: {step_time_ms:.1f}ms | "
                #     f"VRAM: {allocated_mb:.1f}MB" + temp_str
                # )
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = os.path.join(
                        self.args.checkpoint_dir,
                        f"{self.args.model}_{self.args.pipeline}.pt",
                    )
                    torch.save(
                        {
                            "step": step,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_loss": val_loss,
                            "args": self.args,
                            "run_meta": run_meta,
                        },
                        checkpoint_path,
                    )
                    # logger.info(
                    #     f"  [Checkpoint] Saved best {self.args.model} model to {checkpoint_path} (Val Loss: {val_loss:.4f})"
                    # )
                    # ----------------------------------------------------------------
                    # LOGGING SECTION
                    # ----------------------------------------------------------------
                    self.log_metrics(
                        f"{self.args.checkpoint_dir}/{self.args.training_name}/{self.args.model}_{self.args.pipeline}/logs_train.csv",
                        [step, loss_accum, val_ppl, lr, grad_norm.item() ],
                    )
                    self.log_metrics(
                        f"{self.args.checkpoint_dir}/{self.args.training_name}/{self.args.model}_{self.args.pipeline}/logs_validation.csv",
                        [step, val_loss, train_ppl, lr, grad_norm.item() ],
                    )
            step += 1
            # if step % self.args.monitor_interval == 0 or step == steps_per_epoch - 1:
            #     self.log_metrics(
            #         f"{self.args.checkpoint_dir}/{self.args.training_name}/{self.args.model}_{self.args.pipeline}/logs_train.csv",
            #         [step, loss_accum]
            #     )
            #     self.log_metrics(
            #         f"{self.args.checkpoint_dir}/{self.args.training_name}/{self.args.model}_{self.args.pipeline}/logs_validation.csv",
            #         [step, val_loss]
            #     )

        test_loss = self.evaluate_loss(
            model,
            self.args.model,
            self.args.checkpoint_dir,
            Data["test"],
            test_len,
            1,
            self.args.max_seq_len,
            device,
            "PT",
            desc= "Train"
        )
        test_ppl = math.exp(test_loss) if test_loss < 1000 else float("inf")
        print(f"\n TRAIN LOSS: {test_loss} | TRAIN PPL: {test_ppl}")
        total_time_min = (time.time() - t0) / 60
        return total_time_min, best_val_loss
