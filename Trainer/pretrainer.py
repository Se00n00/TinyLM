import logging
import math
import os
import time
from typing import Any, Dict
import asyncio
import queue
import threading
import time

from datasets import load_dataset
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

BUFFER_SIZE = 10_000  # Number of batches kept ready

# How many examples are tokenized together
TOKENIZE_BATCH_SIZE = 64

# Queue waits briefly before retrying when full/empty
QUEUE_TIMEOUT = 1.0
BUFFER = None
STOP_EVENT = threading.Event()

class PRETrainer(Trainer):
    def __init__(self, args, type, data) -> None:
        super().__init__()
        self.args = args
        self.training_type = type
        self.data_config = data
        
        # SETUP MODEL, TOKENIZER, OPTIMIZER
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device.type == "cuda":
            self.enforce_gpu_limit(self.args.vram_limit_mb)

        ptdtype = torch.float16
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=(ptdtype == torch.float16))

        self.tokenizer = AutoTokenizer.from_pretrained("Se00n00/TinyLM-2")
        match self.args.model:
            case "Alibi":
                self.model = Model(Config(vocab_size=len(self.tokenizer)))

            case _:
                self.model = Model(Config(vocab_size=len(self.tokenizer)))

        

        self.args.learning_rate = (
            self.args.learning_rate
            if self.args.learning_rate != 0.0
            else self.get_learning_rate(self.args.pipeline)
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            weight_decay=self.args.weight_decay,
            lr=self.args.learning_rate,
        )

        self.step = 0
        self.best_val_loss = float("inf")
        self.val_loss = float("inf")
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
                    resume_path, map_location=self.device, weights_only=False
                )
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

                # Move optimizer states to target device
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)

                self.step = checkpoint.get("step", 0) + 1
                self.best_val_loss = checkpoint.get("val_loss", float("inf"))
                print(
                    f"Resumed successfully. Continuing from step {self.step} with best val loss: {self.best_val_loss:.4f}"
                )
            else:
                print(
                    f"Warning: Checkpoint path '{resume_path}' not found. Starting training from scratch (step 0)."
                )

    async def train(self):
        if self.training_type == 'huggingface':
            return await self.train_huggingface(self.data_config, self.huggingface_dataset)
        else:
            return await self.train_local(self.data_config)
            
    async def train_huggingface(self, huggingface_dataset:dict, dataset:Dict):
        global BUFFER
        BUFFER = queue.Queue(maxsize=BUFFER_SIZE)
    
        tokenizer = AutoTokenizer.from_pretrained("Se00n00/TinyLM-2")
        
        Data = self.load_dataset_from_local(dataset)
        test_data = Data["test"]["token_path"]
        test_len = len(test_data)
        
        
        BATCH_SIZE = self.args.batch_size  # 8
        GRAD_ACCUM_STEPS = self.args.grad_accum_steps  # 4
        EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRAD_ACCUM_STEPS  # 32
        tokens_per_step = BATCH_SIZE * self.args.max_seq_len * GRAD_ACCUM_STEPS

        num_params = sum(p.numel() for p in self.model.parameters())
        ptdtype = torch.float16
        
        # Save a run metadata JSON
        run_meta = {
            "model": self.args.model,
            "vocab_size": len(self.tokenizer),
            "batch_size": self.args.batch_size,
            "max_seq_len": self.args.max_seq_len,
            "learning_rate": self.args.learning_rate,
        }
        print(
            "\n--------------------------------------------------------------------------"
        )
        print(
            f"DEVICE: {self.device.type} MODEL: {self.args.model}| PARAMETERS:{num_params / (1024 * 1024)}| BATCH SIZE:{BATCH_SIZE}"
        )
        print(
            "----------------------------------------------------------------------------\n"
        )
        # --------------------------------------------------------
        # Start producer
        # --------------------------------------------------------
        
        batch_size = self.args.batch_size
        seq_len = self.args.max_seq_len
        producer_thread = threading.Thread(
            target=self.producer,
            args=(tokenizer, BUFFER, batch_size, seq_len, huggingface_dataset),
            daemon=True,
        )
    
        producer_thread.start()
    
        # --------------------------------------------------------
        # Training / consumer loop
        # --------------------------------------------------------
    
        consumed = 0
        start_time = time.time()
        
        t0 = time.time()
        progress = tqdm(
            unit="batch",
            desc="Training",
            dynamic_ncols=True,
        )
    
        try:
            while True:
                self.check_and_cooldown_gpu(self.args.max_temp, self.args.cooldown_temp)
                lr = get_lr(
                    self.step, steps_per_epoch, self.args.learning_rate, self.args.warmup_steps
                )
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr
                    
                t_start = time.time()
                step_completed = False
                
                while not step_completed:
                    torch.cuda.empty_cache()
                    try:
                        self.optimizer.zero_grad(set_to_none=True)
                        loss_accum = 0.0
    
                        self.check_vram_limit(self.args.vram_limit_mb, self.device)
                        skipped_batch = 0
                        for micro_step in range(GRAD_ACCUM_STEPS):
                            global_micro_step = self.step * GRAD_ACCUM_STEPS + micro_step
                            
                            # queue.get() is blocking, therefore execute it
                            # outside the asyncio event loop.
                            batch = await asyncio.to_thread(BUFFER.get)
                
                            # Producer finished
                            if batch is None:
                                break
                
                            consumed += 1
                            # >
                            batch = self.get_batch(
                                global_micro_step,
                                Data["train"],
                                train_len,
                                BATCH_SIZE,
                                self.args.max_seq_len,
                                self.device,
                                self.args.pipeline,
                            )
    
                            with autocast(device_type=self.device.type, dtype=ptdtype):
                                logits = self.model(batch["inputs"])
                                loss = get_cross_entropy_loss(logits, batch["labels"])
    
                            loss = loss / (GRAD_ACCUM_STEPS - skipped_batch)
    
                            if torch.isnan(loss):
                                print("NaN loss encountered, skipping batch")
                                self.optimizer.zero_grad(set_to_none=True)
                                continue
    
                            loss_accum += loss.item()
    
                            if ptdtype == torch.float16:
                                self.scaler.scale(loss).backward()
                            else:
                                loss.backward()
    
                        if ptdtype == torch.float16:
                            self.scaler.unscale_(self.optimizer)
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), 1.0
                            )
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), 1.0
                            )
                            self.optimizer.step()
    
                        step_completed = True
    
                    except RuntimeError as e:
                        err_msg = str(e).lower()
                        if (
                            "out of memory" in err_msg
                            or "memory limit" in err_msg
                            or "allowed memory" in err_msg
                        ):
                            print(
                                f"\n[Memory Guard] CUDA OOM or limit exceeded at step {self.step} with batch_size={BATCH_SIZE}, grad_accum_steps={GRAD_ACCUM_STEPS}."
                            )
                            self.optimizer.zero_grad(set_to_none=True)
                            torch.cuda.empty_cache()
    
                            if BATCH_SIZE > 1:
                                old_bs = BATCH_SIZE
                                BATCH_SIZE = max(1, BATCH_SIZE // 2)
                                GRAD_ACCUM_STEPS = EFFECTIVE_BATCH_SIZE // BATCH_SIZE
                                print(
                                    f"  [Memory Guard] Halving micro-batch size: {old_bs} -> {BATCH_SIZE}. Increasing grad_accum_steps to {GRAD_ACCUM_STEPS}."
                                )
                            elif not getattr(self.model, "gradient_checkpointing", False):
                                print(
                                    "  [Memory Guard] Micro-batch size is already 1. Enabling gradient checkpointing to save memory..."
                                )
                                self.model.gradient_checkpointing = True
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
                            
    
                        progress.update(1)
            
                        elapsed = time.time() - start_time
            
                        if elapsed > 0:
                            batches_per_second = consumed / elapsed
            
                            progress.set_postfix(
                                {
                                    "buffer": BUFFER.qsize(),
                                    "batch/s": f"{batches_per_second:.2f}",
                                }
                            )
                    
                t_end = time.time()
                step_time_ms = (t_end - t_start) * 1000
    
                # Evaluate
                if self.step % self.args.eval_interval == 0 or self.step == steps_per_epoch - 1:
                    
                    test_loss = self.evaluate_loss(
                        self.model,
                        self.args.model,
                        self.args.checkpoint_dir,
                        Data["test"],
                        test_len,
                        1,
                        self.args.max_seq_len,
                        self.device,
                        "PT",
                        desc= "Test"
                    )
                    test_ppl = math.exp(test_loss) if test_loss < 1000 else float("inf")
                    print(f"\n TEST LOSS: {test_loss} | TEST PPL: {test_ppl}")
                    
                    checkpoint_path = os.path.join(
                        self.args.checkpoint_dir,
                        f"{self.args.training_name}/{self.args.model}_{self.args.pipeline}.pt",
                    )
                    torch.save(
                        {
                            "step": self.step,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "args": self.args,
                            "run_meta": run_meta,
                        },
                        checkpoint_path,
                    )
                    tqdm.write(
                        f"  [Checkpoint] Saved best {self.args.model} model to {checkpoint_path} (Val Loss: {val_loss:.4f})"
                    )
                    
                    # ----------------------------------------------------------------
                    # LOGGING SECTION
                    # ----------------------------------------------------------------
                    train_ppl = math.exp(loss_accum) if loss_accum < 1000 else float("inf")
                    
                    self.log_metrics(
                        f"{self.args.checkpoint_dir}/{self.args.training_name}/{self.args.model}_{self.args.pipeline}/logs_train.csv",
                        [self.step, loss_accum, train_ppl, lr, grad_norm.item() ],
                    )
                    self.log_metrics(
                        f"{self.args.checkpoint_dir}/{self.args.training_name}/{self.args.model}_{self.args.pipeline}/logs_validation.csv",
                        [self.step, test_loss, test_ppl, lr, grad_norm.item() ],
                    )
            self.step += 1
        except asyncio.CancelledError:
            print("\n[TRAINER] Cancellation requested")
    
            STOP_EVENT.set()
    
            raise
    
        finally:
            progress.close()
    
            STOP_EVENT.set()
    
            producer_thread.join(timeout=5)
    
            print(
                f"\nConsumed: {consumed:,} batches | token count: {consumed * batch_size * seq_len}"
            )
    
            print("[TRAINER] Finished")
    
    def batch_generator(self, tokenizer, BATCH_SIZE, SEQ_LEN, PRETRAINING_DATASET):
        """
        Converts the streaming token sequence into fixed
        [BATCH_SIZE, SEQ_LEN] batches.
        """
    
        token_iter = self.token_generator(tokenizer, PRETRAINING_DATASET)
    
        tokens_per_batch = BATCH_SIZE * SEQ_LEN
    
        batch = []
    
        while not STOP_EVENT.is_set():
            try:
                for _ in range(tokens_per_batch):
                    if STOP_EVENT.is_set():
                        return
    
                    batch.append(next(token_iter))
    
            except StopIteration:
                return
    
            if len(batch) != tokens_per_batch:
                return
    
            # uint16 is sufficient for GPT-2's 50,257 vocabulary.
            tensor = torch.tensor(
                batch,
                dtype=torch.uint16,
            ).view(BATCH_SIZE, SEQ_LEN)
    
            batch.clear()
    
            yield tensor
    
    
    # ============================================================
    # BACKGROUND PRODUCER
    # ============================================================
    
    
    def producer(self, tokenizer, buffer, BATCH_SIZE, SEQ_LEN, PRETRAINING_DATASET):
        """
        Runs in a background thread.
    
        Continuously downloads/tokenizes/prepares batches and
        places them into BUFFER.
        """
    
        print("\n[PRODUCER] Started")
    
        batches_produced = 0
        start_time = time.time()
    
        try:
            for batch in self.batch_generator(
                tokenizer, BATCH_SIZE, SEQ_LEN, PRETRAINING_DATASET
            ):
                if STOP_EVENT.is_set():
                    break
    
                # Wait until there is space in the buffer
                while not STOP_EVENT.is_set():
                    try:
                        buffer.put(
                            batch,
                            timeout=QUEUE_TIMEOUT,
                        )
                        break
    
                    except queue.Full:
                        continue
    
                batches_produced += 1
    
                # Print producer throughput occasionally
                if batches_produced % 1000 == 0:
                    elapsed = time.time() - start_time
    
                    if elapsed > 0:
                        throughput = batches_produced / elapsed
    
                        print(
                            f"\n[PRODUCER] "
                            f"{batches_produced:,} batches | "
                            f"{throughput:.2f} batches/s | "
                            f"buffer={buffer.qsize():,}"
                        )
    
        except Exception as e:
            import traceback
            print(f"\n[PRODUCER ERROR] {type(e).__name__}: {e} {traceback.format_exc()}")
    
        finally:
            # Sentinel tells consumer that producer finished
            while not STOP_EVENT.is_set():
                try:
                    buffer.put(
                        None,
                        timeout=QUEUE_TIMEOUT,
                    )
                    break
    
                except queue.Full:
                    continue
    
            print("\n[PRODUCER] Stopped")
    
    def token_generator(self, tokenizer, PRETRAINING_DATASET):
        """
        Streams examples from all datasets and yields token IDs.
    
        Datasets are consumed sequentially.
        """
        
        for dataset_config in PRETRAINING_DATASET:
            if STOP_EVENT.is_set():
                return
    
            kwargs = {
                "path": dataset_config["base"],
                "split": dataset_config["split"],
                "streaming": True,
            }
    
            if dataset_config.get("subset") is not None:
                kwargs["name"] = dataset_config["subset"]
                
            if dataset_config.get("data_dir") is not None:
                kwargs["data_dir"] = dataset_config["data_dir"]
    
            ds = load_dataset(**kwargs)
            total_tokens = 0
    
            for example in ds:
                if total_tokens >= dataset_config['tokens'] and not dataset_config['train_total']:
                    break
                
                if STOP_EVENT.is_set():
                    return
                
                if isinstance(dataset_config['column'], list):
                    text = ''
                    for col in dataset_config['column']:
                        text = text + " " + example.get(col)
                else:
                    text = example.get(dataset_config['column'])
                
    
                if text is None:
                    continue
    
                # Tokenize one document
                tokens = tokenizer.encode(
                    text,
                    tokenize=True,
                    add_generation_prompt=False
                )
                
                for token in tokens:
                    total_tokens += 1
                    yield token
            
            print(f"TRAINED: {dataset_config['base']} | TOTAL TOKENS: {total_tokens}")
            
    async def train_local(
        self, dataset: Dict
    )-> tuple:
        super().train()

        # 1. Load / Setup Dataset
        Data = self.load_dataset_from_local(dataset)
        # Else setup Streaming for Pre-training using hugginface dataset

        
        self.model.to(self.device)
        self.model.train()

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
        num_params = sum(p.numel() for p in self.model.parameters())
        ptdtype = torch.float16
        
        # Save a run metadata JSON
        run_meta = {
            "model": self.args.model,
            "vocab_size": len(self.tokenizer),
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
            f"DEVICE: {self.device.type} MODEL: {self.args.model}| PARAMETERS:{num_params / (1024 * 1024)}| BATCH SIZE:{BATCH_SIZE}"
        )
        print(
            "----------------------------------------------------------------------------\n"
        )
        
        
        t0 = time.time()
        pbar1 = tqdm(total=steps_per_epoch, desc="Train")
        while self.step < steps_per_epoch:
            self.check_and_cooldown_gpu(self.args.max_temp, self.args.cooldown_temp)
            lr = get_lr(
                self.step, steps_per_epoch, self.args.learning_rate, self.args.warmup_steps
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            t_start = time.time()
            step_completed = False

            while not step_completed:
                torch.cuda.empty_cache()

                try:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss_accum = 0.0

                    self.check_vram_limit(self.args.vram_limit_mb, self.device)
                    skipped_batch = 0
                    for micro_step in range(GRAD_ACCUM_STEPS):
                        global_micro_step = self.step * GRAD_ACCUM_STEPS + micro_step

                        batch = self.get_batch(
                            global_micro_step,
                            Data["train"],
                            train_len,
                            BATCH_SIZE,
                            self.args.max_seq_len,
                            self.device,
                            self.args.pipeline,
                        )

                        with autocast(device_type=self.device.type, dtype=ptdtype):
                            logits = self.model(batch["inputs"])
                            loss = get_cross_entropy_loss(logits, batch["labels"])

                        loss = loss / (GRAD_ACCUM_STEPS - skipped_batch)

                        if torch.isnan(loss):
                            print("NaN loss encountered, skipping batch")
                            self.optimizer.zero_grad(set_to_none=True)
                            continue

                        loss_accum += loss.item()

                        if ptdtype == torch.float16:
                            self.scaler.scale(loss).backward()
                        else:
                            loss.backward()

                    if ptdtype == torch.float16:
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 1.0
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 1.0
                        )
                        self.optimizer.step()

                    step_completed = True

                except RuntimeError as e:
                    err_msg = str(e).lower()
                    if (
                        "out of memory" in err_msg
                        or "memory limit" in err_msg
                        or "allowed memory" in err_msg
                    ):
                        print(
                            f"\n[Memory Guard] CUDA OOM or limit exceeded at step {self.step} with batch_size={BATCH_SIZE}, grad_accum_steps={GRAD_ACCUM_STEPS}."
                        )
                        self.optimizer.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()

                        if BATCH_SIZE > 1:
                            old_bs = BATCH_SIZE
                            BATCH_SIZE = max(1, BATCH_SIZE // 2)
                            GRAD_ACCUM_STEPS = EFFECTIVE_BATCH_SIZE // BATCH_SIZE
                            print(
                                f"  [Memory Guard] Halving micro-batch size: {old_bs} -> {BATCH_SIZE}. Increasing grad_accum_steps to {GRAD_ACCUM_STEPS}."
                            )
                        elif not getattr(self.model, "gradient_checkpointing", False):
                            print(
                                "  [Memory Guard] Micro-batch size is already 1. Enabling gradient checkpointing to save memory..."
                            )
                            self.model.gradient_checkpointing = True
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
            if self.step % self.args.eval_interval == 0 or self.step == steps_per_epoch - 1:
                val_loss = self.evaluate_loss(
                    self.model,
                    self.args.model,
                    self.args.checkpoint_dir,
                    Data["validation"],
                    val_len,
                    1,
                    self.args.max_seq_len,
                    self.device,
                    self.args.pipeline,
                )
                val_ppl = math.exp(val_loss) if val_loss < 1000 else float("inf")
                train_ppl = math.exp(loss_accum) if loss_accum < 1000 else float("inf")
                
                allocated_mb = (
                    torch.cuda.memory_allocated(self.device) / (1024 * 1024)
                    if self.device.type == "cuda"
                    else 0.0
                )
                temp_str = (
                    f" | Temp: {self.get_gpu_temperature()}°C"
                    if self.get_gpu_temperature() is not None
                    else ""
                )
                tqdm.write(
                    f"Step {self.step:4d}/{steps_per_epoch:4d} | "
                    f"Train Loss: {self.loss_accum:.4f} (PPL: {train_ppl:.2f}) | "
                    f"Val Loss: {val_loss:.4f} (PPL: {val_ppl:.2f}) | "
                    f"LR: {lr:.2e} | "
                    f"Grad Norm: {self.grad_norm:.2f} | "
                    f"Step Time: {step_time_ms:.1f}ms | "
                    f"VRAM: {allocated_mb:.1f}MB" + temp_str
                )
                
                if val_loss < self.best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = os.path.join(
                        self.args.checkpoint_dir,
                        f"{self.args.training_name}/{self.args.model}_{self.args.pipeline}.pt",
                    )
                    torch.save(
                        {
                            "step": self.step,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "val_loss": val_loss,
                            "args": self.args,
                            "run_meta": run_meta,
                        },
                        checkpoint_path,
                    )
                    tqdm.write(
                        f"  [Checkpoint] Saved best {self.args.model} model to {checkpoint_path} (Val Loss: {val_loss:.4f})"
                    )
                    # ----------------------------------------------------------------
                    # LOGGING SECTION
                    # ----------------------------------------------------------------
                    self.log_metrics(
                        f"{self.args.checkpoint_dir}/{self.args.training_name}/{self.args.model}_{self.args.pipeline}/logs_train.csv",
                        [self.step, loss_accum, train_ppl, lr, grad_norm.item() ],
                    )
                    self.log_metrics(
                        f"{self.args.checkpoint_dir}/{self.args.training_name}/{self.args.model}_{self.args.pipeline}/logs_validation.csv",
                        [self.step, val_loss, val_ppl, lr, grad_norm.item() ],
                    )
            self.step += 1
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
            self.model,
            self.args.model,
            self.args.checkpoint_dir,
            Data["test"],
            test_len,
            1,
            self.args.max_seq_len,
            self.device,
            "PT",
            desc= "Test"
        )
        test_ppl = math.exp(test_loss) if test_loss < 1000 else float("inf")
        print(f"\n TEST LOSS: {test_loss} | TEST PPL: {test_ppl}")
        total_time_min = (time.time() - t0) / 60
        return total_time_min, self.best_val_loss
