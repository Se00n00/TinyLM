"""
CHAT TEMPLATES
-------------------------
For Text Genration
-------------------------
"messages": {"text": ...}
-------------------------
Instruction Fine Tunning
-------------------------
"messages":[
    {
        "role":"system",
        "content":"..."
    },
    {
        "role":"user",
        "content":"..."
    },
    {
        "role":"assistant",
        "content": "..."
    }
]
-------------------------
Reasoning Fine Tunning
-------------------------
"messages":[
    {
        "role":"system",
        "content":"..."
    },
    {
        "role":"user",
        "content":"..."
    },
    {
        "role":"assistant",
        "content": "<|THINK|> ... <|THINK|> ..."
    }
]
-------------------------
Tool Calling
-------------------------
"messages":[{
    "available_tools": [
        {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Name of the city."
                    }
                },
                "required": ["city"]
            }
        }
    ],
    "system": "...",
    "user":"...",
    "assisstant": ".. <|TOOL_CALLS|>[
        {
            "name": "...",
            "arguments":{
                "expression":"..."
            }
        }
    ]<|/TOOL_CALLS|> ..."
}]

---------------------------
Tool Calling + Reasoning
---------------------------
"messages":[
    {
        "role":"available_tools",
        "content":[
            {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "Name of the city."
                        }
                    },
                    "required": ["city"]
                }
            }
        ]
    },
    {
        "role": "system",
        "content":"...",
    },
    {
        "role": "user",
        "content": "..."
    },
    {
        "role":"assistant",
        "content": "<|THINK|> ... <|/THINK|> ... <|TOOL_CALLS|>[
            {
                "name": "...",
                "arguments":{
                    "expression":"..."
                }
            }
        ]<|/TOOL_CALLS|> ... "
    }
]

"""

import csv
import math
import os
import time
from ast import Pass
from dataclasses import dataclass
from pathlib import Path
from datetime import timedelta
import datasets
import torch
import torch.nn.functional as F
from datasets.arrow_dataset import Dataset
from datasets.iterable_dataset import IterableDataset as HFIterableDataset
from torch.amp import autocast
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset as TorchIterableDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import torch.distributed as dist

from .base import Trainer
from .sftconfig import SFTConfig
from .util import get_lr


class SFTTrainer(Trainer):
    def __init__(
        self,
        training_name,
        current_example,
        model,
        tokenizer,
        ds: Dataset | HFIterableDataset,
        config: SFTConfig,
    ):
        self.training_name = training_name
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.current_example = current_example
        (
            self.is_ddp,
            self.rank,
            self.local_rank,
            self.world_size,
        ) = self._setup_ddp(config)
        self.is_main_process = (not self.is_ddp) or self.rank == 0
        
        if self.is_ddp:
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(ds, Dataset):
            Data = ds.train_test_split(test_size=config.test_train_ratio)
            train_data = Data["train"]
            test_data = Data["test"]
            self.test_samples = test_data.num_rows
            self.iterable_train_data = False

        else:
            self.test_samples = int(config.total_samples * config.test_train_ratio)
            self.train_samples = config.total_samples - self.test_samples
            train_data = ds.skip(self.test_samples)
            test_ = ds.take(self.test_samples)

            test_data = Dataset.from_generator(lambda: (item for item in test_))

            self.iterable_train_data = True
        
        if self.is_ddp:
            train_data = train_data.shard(
                num_shards=self.world_size, index=self.rank, contiguous=True
            )
            test_data = test_data.shard(
                num_shards=self.world_size, index=self.rank, contiguous=True
            )
        self.train_data = train_data
        self.test_data = test_data

        model.to(self.device)
         
        if self.is_ddp:
            self.model = DDP(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=config.ddp_find_unused_parameters,
            )
            # Use this whenever you need the *underlying* model - e.g.
            # state_dict()/load_state_dict() (to avoid the "module."
            # prefix DDP adds) or custom attributes like
            # `gradient_checkpointing`.
            self.raw_model = self.model.module
        else:
            self.model = model
            self.raw_model = model
            
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            weight_decay=self.config.weight_decay,
            lr=self.config.learning_rate,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = torch.amp.GradScaler(
            self.device.type, enabled=(self.config.ptdtype == torch.float16)
        )

        # Initiallize .csv files for Logging
        if self.is_main_process:
            self.init_csv(
                f"{config.checkpoint_dir}/{training_name}/logs_train.csv",
                [
                    "step",
                    "num_examples",
                    "loss",
                    "perplexity",
                    "entropy",
                    "mean_token_accuracy",
                    "learning_rate",
                    "GNorm",
                ],
            )
            self.init_csv(
                f"{config.checkpoint_dir}/{training_name}/logs_validation.csv",
                [
                    "step",
                    "num_examples",
                    "loss",
                    "perplexity",
                    "entropy",
                    "mean_token_accuracy",
                ],
            )
    
        # Make sure rank 0 has finished creating the checkpoint dir/csvs
        # before anyone else proceeds (matters mostly for `resume="auto"`
        # right after a fresh checkpoint dir is created).
        if self.is_ddp:
            dist.barrier()

    # -----------------------------------------------------------------
    # DDP HELPERS
    # -----------------------------------------------------------------
    @staticmethod
    def _setup_ddp(config: SFTConfig):
        """
        Initializes the default process group if launched via torchrun.
        Returns (is_ddp, rank, local_rank, world_size).
        """
        if "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) <= 1:
            return False, 0, 0, 1
    
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    
        backend = config.ddp_backend if torch.cuda.is_available() else "gloo"
    
        if not dist.is_initialized():
            torch.cuda.set_device(local_rank)
            device = (
                torch.device("cuda", local_rank)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            dist.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size,
                timeout=timedelta(seconds=config.ddp_timeout_seconds),
                device_id=device,
            )
    
        return True, rank, local_rank, world_size
    
    def _cleanup_ddp(self):
        if self.is_ddp and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
    
    def _reduce_mean(self, value: float) -> float:
        """
        Averages a python float across all ranks. Used purely for
        *logging/printing* a globally representative loss/entropy/accuracy
        number - it does not affect gradients, which DDP already
        synchronizes (averages) internally during backward().
        """
        if not self.is_ddp:
            return value
    
        t = torch.tensor(value, dtype=torch.float64, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return (t / self.world_size).item()
    
    def _reduce_sum(self, value) -> float:
        """
        Sums a python number across all ranks. Used for turning each
        rank's local "how many rows have I consumed from my shard"
        counter into a single global row count, e.g. for checkpointing.
        """
        if not self.is_ddp:
            return float(value)
    
        t = torch.tensor(float(value), dtype=torch.float64, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t.item()
        
    def _barrier(self):
        if self.is_ddp:
            dist.barrier()
    
    # -----------------------------------------------------------------
    # LAUNCH HELPERS (mp.spawn, for when you don't want to use torchrun)
    # -----------------------------------------------------------------
    @staticmethod
    def launch(world_size: int, master_addr="localhost", master_port="29500", **trainer_kwargs):
        """
        Alternative to `torchrun` - spawns `world_size` local processes
        with `torch.multiprocessing`, each running its own SFTTrainer,
        and calls `.train()` on each. Use this when you want a single
        `python train.py` entrypoint to fan out to multiple GPUs itself,
        instead of invoking `torchrun --nproc_per_node=...` externally.
    
        `trainer_kwargs` are exactly the kwargs you'd otherwise pass to
        `SFTTrainer(...)` (training_name, model, tokenizer, ds, config,
        current_example). They get pickled and sent to each spawned
        process, so:
            - `model` should still be on CPU (not `.to(device)` yet) -
            SFTTrainer.__init__ moves it to the right device per rank.
            - `ds`, if it's a streaming HF IterableDataset, must be
            picklable - this holds for the normal
            `load_dataset(..., streaming=True)` result, but custom
            generators/closures inside it may not survive pickling.
    
        Prefer `torchrun` for multi-node training - `mp.spawn` here only
        covers single-node, multi-GPU.
        """
        import torch.multiprocessing as mp
    
        os.environ.setdefault("MASTER_ADDR", master_addr)
        os.environ.setdefault("MASTER_PORT", str(master_port))
    
        mp.spawn(
            SFTTrainer._mp_entry,
            args=(world_size, trainer_kwargs),
            nprocs=world_size,
            join=True,
        )
    
    @staticmethod
    def _mp_entry(local_rank, world_size, trainer_kwargs):
        # mp.spawn gives us `local_rank` directly. Since this helper only
        # supports single-node launches, global rank == local rank and
        # world_size is exactly what was requested.
        os.environ["RANK"] = str(local_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(world_size)
    
        trainer = SFTTrainer(**trainer_kwargs)
        trainer.train()
        
    # -----------------------------------------------------------------
    # RESUME / DATASET-SKIP HELPERS
    # -----------------------------------------------------------------
    @staticmethod
    def peek_checkpoint(config: SFTConfig, training_name: str):
        """
        Reads just the bookkeeping fields out of a checkpoint (step,
        current_example, val_loss) WITHOUT touching the model - meant to
        be called from your training script, BEFORE constructing the
        dataset, so you can `.skip(...)` past already-consumed rows on
        resume. Returns None if `config.resume` is falsy or the
        checkpoint file doesn't exist yet.
    
        Example:
            state = SFTTrainer.peek_checkpoint(config, training_name)
            if state is not None:
                ds = ds.skip(state["current_example"])
        """
        if not config.resume:
            return None
    
        resume_path = config.resume
        if resume_path.lower() == "auto":
            resume_path = os.path.join(config.checkpoint_dir, f"{training_name}/model.pt")
    
        if not os.path.exists(resume_path):
            return None
    
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        return {
            "step": checkpoint.get("step", 0),
            "current_example": checkpoint.get("current_example", 0),
            "val_loss": checkpoint.get("val_loss", float("inf")),
        }
  
        
    def init_csv(self, csv_path, cols):
        if not Path(csv_path).exists():
            Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(cols)
    
    def train(self):
        super().train()
        try:
            self._train_loop()
        finally:
            self._cleanup_ddp()

    def _train_loop(self):
            self.best_val_loss = float("inf")
            BATCH_SIZE = self.config.batch_size
            GRAD_ACCUM_STEPS = self.config.grad_accum_steps
            EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRAD_ACCUM_STEPS
    
            num_params = sum(p.numel() for p in self.model.parameters())
    
            step = 0
            loss_accum = float("inf")
            lr = 0
            grad_norm = torch.tensor(0.0)
            entropy, mean_token_accuracy = 0, 0
    
            if self.config.resume:
                resume_path = self.config.resume
    
                if resume_path.lower() == "auto":
                    resume_path = os.path.join(
                        self.config.checkpoint_dir,
                        f"{self.training_name}/model.pt",
                    )
    
                if os.path.exists(resume_path):
                    # map_location pins the checkpoint tensors to *this*
                    # rank's device, so every rank loads its own local copy
                    # rather than everyone deserializing onto rank 0's GPU.
                    checkpoint = torch.load(
                        resume_path, map_location=self.device, weights_only=False
                    )
                    self.raw_model.load_state_dict(checkpoint["model_state_dict"])
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(self.device)
    
                    step = checkpoint.get("step", 0) + 1
                    if self.config.resum_same_dataset == False:
                        step = 0
                    else:
                        # Continuing on the same dataset: restore how far
                        # into it we were (this drives both the LR schedule
                        # below and, more importantly, is what your training
                        # script should have used to `.skip(...)` the raw
                        # dataset via `SFTTrainer.peek_checkpoint(...)`
                        # BEFORE it was ever passed into this trainer -
                        # otherwise you'll be training over the same
                        # examples again from the start of the stream.
                        self.current_example = checkpoint.get(
                            "current_example", self.current_example
                        )
                        
                        self.current_example =  self.current_example // self.world_size 
    
                    self.best_val_loss = checkpoint.get("val_loss", float("inf"))
    
                else:
                    if self.is_main_process:
                        print(
                            f"Warning: Checkpoint path '{resume_path}' not found. Starting training from scratch (step 0)."
                        )
    
                # Every rank must agree on `step`/`best_val_loss` before
                # continuing, otherwise the LR schedule (which depends on
                # `step`/`current_example`) could desync across ranks.
                self._barrier()
    
            consumed = 0
            t_initial = time.time()
            
            total_iterations = (
                self.config.total_samples // self.world_size
                if self.is_ddp
                else self.config.total_samples
            )
            train_bar = tqdm(
                total=total_iterations, desc="Training", unit="ex", dynamic_ncols=True, disable=not self.is_main_process
            )
    
            try:
    
                if self.config.resum_same_dataset == False and self.config.resume == True:
                    self.current_example = int(
                        total_iterations * self.config.warmup_steps_ratio
                    )
    
                run_meta = {
                    "vocab_size": len(self.tokenizer),
                    "batch_size": self.config.batch_size,
                    "max_seq_len": self.config.max_length,
                    "learning_rate": self.config.learning_rate,
                }
                if self.is_main_process:
                    print(
                        "\n--------------------------------------------------------------------------"
                    )
                    print(
                        f"DEVICE: {self.device.type} | PARAMETERS:{num_params / (1024 * 1024)}| "
                        f"BATCH SIZE:{BATCH_SIZE} | WORLD SIZE: {self.world_size}"
                    )
                    print(
                        "----------------------------------------------------------------------------\n"
                    )
                # ---------------------------------------------
                # TRAIN LOOP
                # ---------------------------------------------
    
                Batches = iter(
                    self.get_batch(self.train_data, streaming=self.iterable_train_data)
                )
                EPOCH_COMPLETED = False
                while not EPOCH_COMPLETED:
                    # NOTE on DDP + GPU-temperature cooldown: this check runs
                    # per-rank against *that rank's own* GPU. If ranks are on
                    # different physical machines/cards with different
                    # thermals, one rank could sleep here while others don't,
                    # which risks a collective-op timeout during the next
                    # backward() all-reduce. Set config.ddp_sync_cooldown=False
                    # if you'd rather accept that risk than always throttle
                    # every rank to the slowest/hottest one; True (default)
                    # is a stronger guard but adds a barrier.
                    self.check_and_cooldown_gpu(
                        self.config.max_temp, self.config.cooldown_temp
                    )
                    if self.is_ddp and self.config.ddp_sync_cooldown:
                        self._barrier()
    
                    lr = get_lr(
                        step,
                        total_iterations,
                        self.config.learning_rate,
                        int(total_iterations * self.config.warmup_steps_ratio) if self.config.warmup_steps_ratio > 0 else 2000,
                        self.config.min_lr_ratio,
                    )
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = lr
    
                    t_start = time.time()
                    step_completed = False
    
                    while not step_completed:
                        torch.cuda.empty_cache()
                        self.optimizer.zero_grad(set_to_none=True)
                        
                        loss_accum = 0.0
                        entropy_accum = 0.0
                        accuracy_accum = 0.0
                        valid_micro_steps = 0
    
                        self.check_vram_limit(self.config.vram_limit_mb, self.device)
                        
                        for micro_step in range(GRAD_ACCUM_STEPS):
                            global_micro_step = step * GRAD_ACCUM_STEPS + micro_step
    
                            batch = next(Batches, None)
                            if batch is None:
                                EPOCH_COMPLETED = True
                                break
    
                            with autocast(
                                device_type=self.device.type, dtype=self.config.ptdtype
                            ):
                                batch_x = batch["data"].to(self.device, non_blocking=True)
                                batch_y = batch["labels"].to(self.device, non_blocking=True)
                                
                                valid_tokens = (batch_y != self.config.label_idx).sum()
                                
                                if valid_tokens.item() == 0:
                                    continue
                                
                                valid_micro_steps += 1
                                
                                # print("valid target tokens:", valid_tokens.item())
                                # print("labels shape:", batch_y.shape)
    
                                logits = self.model(batch_x)
                                raw_loss = F.cross_entropy(
                                    logits.view(-1, logits.size(-1)),
                                    batch_y.view(-1),
                                    ignore_index=self.config.label_idx,
                                )
                                batch_entropy, batch_accuracy = (
                                    self.get_entropy_and_mean_token_accuracy(
                                        logits, batch_y, self.config.label_idx
                                    )
                                )
                                
                                # print("loss:", raw_loss)
                                # print("requires_grad:", raw_loss.requires_grad)
                                # print("grad_fn:", raw_loss.grad_fn)
    
                            if not torch.isfinite(raw_loss):
                                if self.is_main_process:
                                    print("Non-finite loss encountered.")
                                self.optimizer.zero_grad(set_to_none=True)
                                step_completed = False
                                break
    
                            loss_accum += raw_loss.item()
                            entropy_accum += batch_entropy
                            accuracy_accum += batch_accuracy
    
                            loss = raw_loss / GRAD_ACCUM_STEPS
    
                            # DDP all-reduces gradients across ranks every time
                            # `.backward()` is called. During grad-accumulation
                            # we only actually want that sync on the *final*
                            # micro-step of the accumulation window - doing it
                            # on every micro-step wastes a lot of network
                            # bandwidth for no benefit. `model.no_sync()`
                            # disables the automatic all-reduce for all but
                            is_last_micro_step = (
                                valid_micro_steps == GRAD_ACCUM_STEPS
                                or micro_step == GRAD_ACCUM_STEPS - 1
                            )
                        
                            sync_context = (
                                self.model.no_sync()
                                if self.is_ddp and not is_last_micro_step
                                else _nullcontext()
                            )# the last micro-step.
                            
                            # is_last_micro_step = micro_step == GRAD_ACCUM_STEPS - 1
                            
                            
    
                            with sync_context:
                                if self.config.ptdtype == torch.float16:
                                    self.scaler.scale(loss).backward()
                                else:
                                    loss.backward()
    
                            consumed += 1
    
                        loss_accum /= GRAD_ACCUM_STEPS
                        entropy = entropy_accum / GRAD_ACCUM_STEPS
                        mean_token_accuracy = accuracy_accum / GRAD_ACCUM_STEPS
    
                        if self.config.ptdtype == torch.float16:
                            self.scaler.unscale_(self.optimizer)
    
                        # Computed on local (post-all-reduce) gradients, which
                        # are already identical across ranks by this point, so
                        # this grad_norm is already the "global" norm - no
                        # further reduction needed here.
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            1.0,
                        )
    
                        if self.config.ptdtype == torch.float16:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()
    
                        step_completed = True
    
                    if self.is_main_process:
                        train_bar.n = min(self.current_example, total_iterations)
                        train_bar.refresh()

                    elapsed = time.time() - t_start
    
                    if self.is_main_process:
                        postfix = {"remaining": max(0, total_iterations - self.current_example)}
                        if elapsed > 0:
                            postfix["batch/s"] = f"{consumed / elapsed:.2f}"
                        train_bar.set_postfix(postfix)
    
                    # -------------------------------------------------------
                    # EVAL STEP/
                    # -------------------------------------------------------
                    if step % self.config.eval_steps == 0 or (EPOCH_COMPLETED == True):
                        self.model.eval()
                        val_loss = 0.0
                        val_entropy, val_mean_token_accuracy = 0.0, 0.0
                    
                        if self.is_main_process:
                            total_eval_loss = 0.0
                            eval_steps = 0
                    
                            for batch in tqdm(
                                self.get_batch(self.test_data, streaming=False, count_examples=False),
                                desc="Validation",
                                total=len(self.test_data),
                                unit="ex",
                            ):
                                batch_x = batch["data"].to(self.device, non_blocking=True)
                                batch_y = batch["labels"].to(self.device, non_blocking=True)
                    
                                with autocast(device_type=self.device.type, dtype=torch.float16):
                                    logits = self.raw_model(batch_x)
                                    loss = F.cross_entropy(
                                        logits.view(-1, logits.size(-1)),
                                        batch_y.view(-1),
                                        ignore_index=self.config.label_idx,
                                    )
                                    batch_entropy, batch_mean_token_accuracy = (
                                        self.get_entropy_and_mean_token_accuracy(
                                            logits, batch_y, self.config.label_idx
                                        )
                                    )
                                    val_entropy += batch_entropy
                                    val_mean_token_accuracy += batch_mean_token_accuracy
                    
                                total_eval_loss += loss.item()
                                eval_steps += 1
                    
                            eval_steps = max(eval_steps, 1)
                            val_entropy = val_entropy / eval_steps
                            val_mean_token_accuracy = val_mean_token_accuracy / eval_steps
                            val_loss = total_eval_loss / eval_steps
                    
                        global_current_example = self._reduce_sum(self.current_example)
                    
                        val_ppl = math.exp(val_loss) if val_loss < 1000 else float("inf")
                        train_ppl = math.exp(loss_accum) if loss_accum < 1000 else float("inf")
                        self.model.train()
    
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
    
                        if self.is_main_process:
                            tqdm.write(
                                f"Step {step:4d} | "
                                f"Train Loss: {loss_accum:.4f} (PPL: {train_ppl:.2f}) | "
                                f"Val Loss: {val_loss:.4f} (PPL: {val_ppl:.2f}) | "
                                f"LR: {lr:.2e} | "
                                f"Grad Norm: {grad_norm:.2f} | "
                                f"Step Time: {elapsed * 1000:.1f}ms | "
                                f"VRAM: {allocated_mb:.1f}MB" + temp_str
                            )
    
                            if val_loss < self.best_val_loss:
                                self.best_val_loss = val_loss
                                checkpoint_path = os.path.join(
                                    self.config.checkpoint_dir,
                                    f"{self.training_name}/best.pt",
                                )
                                torch.save(
                                    {
                                        "step": step,
                                        "val_loss": val_loss,
                                        # Global row count consumed so far -
                                        # see `SFTTrainer.peek_checkpoint`.
                                        "current_example": int(global_current_example),
                                        # Save the *unwrapped* model so the
                                        # checkpoint loads cleanly whether or
                                        # not it's later resumed under DDP.
                                        "model_state_dict": self.raw_model.state_dict(),
                                        "optimizer_state_dict": self.optimizer.state_dict(),
                                        "run_meta": run_meta,
                                    },
                                    checkpoint_path,
                                )
                                tqdm.write(
                                    f"  [Checkpoint] Saved best model to {checkpoint_path} (Val Loss: {val_loss:.4f})"
                                )
    
                            # TRAINING LOGGING: STEP, LOSS, PERPLEXITY, ENTROPY, MEAN_TOKEN_ACCURACY
                            self.log_metrics(
                                f"{self.config.checkpoint_dir}/{self.training_name}/logs_validation.csv",
                                [
                                    step,
                                    self.current_example
                                    * BATCH_SIZE
                                    * self.config.max_length
                                    * self.world_size,
                                    val_loss,
                                    val_ppl,
                                    val_entropy,
                                    val_mean_token_accuracy,
                                ],
                            )
    
                        # Make sure no rank races ahead into more training
                        # steps while rank 0 is still writing the checkpoint.
                        self._barrier()
    
                    # TRAINING LOGGING: STEP, NUM_TOKENS, LOSS, PERPLEXITY, ENTROPY, MEAN_TOKEN_ACCURACY, LR, GRAD_NORM
                    if (
                        step % self.config.logging_steps == 0 or (EPOCH_COMPLETED == True)
                    ) and self.is_main_process:
                        train_ppl = (
                            math.exp(loss_accum) if loss_accum < 1000 else float("inf")
                        )
    
                        self.log_metrics(
                            f"{self.config.checkpoint_dir}/{self.training_name}/logs_train.csv",
                            [
                                step,
                                self.current_example
                                * BATCH_SIZE
                                * self.config.max_length
                                * self.world_size,
                                loss_accum,
                                train_ppl,
                                entropy,
                                mean_token_accuracy,
                                lr,
                                grad_norm.item(),
                            ],
                        )
                        
                        checkpoint_path = os.path.join(
                            self.config.checkpoint_dir,
                            f"{self.training_name}/model.pt",
                        )
                        torch.save(
                            {
                                "step": step,
                                # Global row count consumed so far -
                                # see `SFTTrainer.peek_checkpoint`.
                                "current_example": self._reduce_sum(self.current_example),
                                # Save the *unwrapped* model so the
                                # checkpoint loads cleanly whether or
                                # not it's later resumed under DDP.
                                "model_state_dict": self.raw_model.state_dict(),
                                "optimizer_state_dict": self.optimizer.state_dict(),
                                "run_meta": run_meta,
                            },
                            checkpoint_path,
                        )
                    self._barrier()
    
                    step += 1
                if self.is_main_process:
                    print("\nTRAINING COMPLETE !")
            except RuntimeError as e:
                tqdm.write(str(e))
                err_msg = str(e).lower()
                if (
                    "out of memory" in err_msg
                    or "memory limit" in err_msg
                    or "allowed memory" in err_msg
                ):
                    if self.is_main_process:
                        print(
                            f"\n[Memory Guard] CUDA OOM or limit exceeded at step {step} with batch_size={BATCH_SIZE}, grad_accum_steps={GRAD_ACCUM_STEPS}."
                        )
                    self.optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
    
                    # IMPORTANT under DDP: if only *one* rank hits OOM while
                    # the others don't, they will diverge on batch size /
                    # grad_accum_steps below and the next collective op
                    # (gradient all-reduce) will hang or error out, since
                    # ranks must call the exact same sequence of collectives
                    # in lockstep. This basic recovery logic is kept
                    # per-rank for simplicity/parity with the original
                    # single-GPU version - for production multi-GPU use,
                    # consider detecting OOM via an all-reduce'd flag so
                    # every rank shrinks batch size together.
                    if BATCH_SIZE > 1:
                        old_bs = BATCH_SIZE
                        BATCH_SIZE = max(1, BATCH_SIZE // 2)
                        GRAD_ACCUM_STEPS = EFFECTIVE_BATCH_SIZE // BATCH_SIZE
                        if self.is_main_process:
                            print(
                                f"  [Memory Guard] Halving micro-batch size: {old_bs} -> {BATCH_SIZE}. Increasing grad_accum_steps to {GRAD_ACCUM_STEPS}."
                            )
                    elif not getattr(self.raw_model, "gradient_checkpointing", False):
                        if self.is_main_process:
                            print(
                                "  [Memory Guard] Micro-batch size is already 1. Enabling gradient checkpointing to save memory..."
                            )
                        self.raw_model.gradient_checkpointing = True
                        # Reset batch size to original/default to try to recover with checkpointing
                        BATCH_SIZE = self.config.batch_size
                        GRAD_ACCUM_STEPS = self.config.grad_accum_steps
                    else:
                        if self.is_main_process:
                            print(
                                "  [Memory Guard] Out of memory even with micro-batch size 1 and gradient checkpointing active."
                            )
    
                        train_bar.close()
                        raise e
                else:
                    train_bar.close()
                    raise e
    
    def get_batch(self, DATASET, streaming=True, count_examples=True):
        """
        Creates a DataLoader for SFT training.

        Expected dataset examples can be either:

            {
                "messages": [
                    {"role": "user", "content": "..."},
                    {"role": "assistant", "content": "..."}
                ]
            }

        or, if chat_template is not being used:

            {
                "text": "..."
            }

        Returns:
            DataLoader yielding:
                {
                    "data":   Tensor[B, max_length],
                    "labels": Tensor[B, max_length]
                }

        SFT behavior:
            - tight packing
            - causal LM shifted labels
            - non-assistant tokens -> -100
            - padding -> -100

        NOTE (DDP): `DATASET` passed in here is already the per-rank
        shard produced in __init__ (self.train_data / self.test_data),
        so no further sharding happens in this method - each rank simply
        iterates its own slice independently.
        """
        max_length = self.config.max_length
        batch_size = self.config.batch_size

        tokenizer = self.tokenizer

        def encode_example(example):
            messages = example.get("messages", None)

            # -----------------------------------------------------
            # Chat dataset
            # -----------------------------------------------------
            if messages is not None:
                if hasattr(tokenizer, "apply_chat_template"):
                    # We need the assistant boundaries.
                    #
                    # The cleanest way is to tokenize the complete
                    # conversation and separately tokenize the prefix
                    # before the assistant response.
                    #
                    # This assumes normal SFT data:
                    # user -> assistant

                    tools = next(
                        (
                            msg.get("content")
                            for msg in messages
                            if msg.get("role") == "available_tools"
                        ),
                        None,
                    )

                    chat_messages = [
                        msg for msg in messages if msg.get("role") != "available_tools"
                    ]
                    tokens = tokenizer.apply_chat_template(
                        chat_messages,
                        tokenize=True,
                        tools=tools,
                        add_generation_prompt=False,
                    )["input_ids"]

                    assistant_mask = [0] * len(tokens)

                    # Find assistant messages and mark their content.
                    #
                    # We construct prefixes so we know exactly which
                    # tokens belong to each assistant response.
                    for i, msg in enumerate(chat_messages):
                        if msg.get("role") != "assistant":
                            continue

                        # Conversation before this assistant message

                        prefix_messages = chat_messages[:i]
                        prefix_tokens = tokenizer.apply_chat_template(
                            prefix_messages,
                            tokenize=True,
                            tools=tools,
                            add_generation_prompt=True,
                        )

                        # Tokenize the assistant content itself.
                        content_tokens = tokenizer(
                            msg.get("content", ""),
                            add_special_tokens=False,
                        )["input_ids"]

                        start = len(prefix_tokens["input_ids"])

                        # Depending on the tokenizer's chat template,
                        # there may be assistant header/special tokens
                        # before the actual content.
                        #
                        # Mark the content tokens only.
                        end = min(
                            start + len(content_tokens),
                            len(assistant_mask),
                        )

                        assistant_mask[start:end] = [1] * (end - start)

                else:
                    # Fallback if tokenizer doesn't have a chat template.
                    text = ""

                    for msg in messages:
                        text += f"{msg['role']}: {msg['content']}\n"

                    encoded = tokenizer(
                        text,
                        add_special_tokens=True,
                    )

                    tokens = encoded["input_ids"]

                    # No reliable assistant boundaries here.
                    # Therefore don't train on this path as SFT unless
                    # you provide your own masking logic.
                    assistant_mask = [1] * len(tokens)

            # -----------------------------------------------------
            # Plain text dataset
            # -----------------------------------------------------
            elif "text" in example:
                encoded = tokenizer(
                    example["text"],
                    add_special_tokens=True,
                )

                tokens = encoded["input_ids"]
                assistant_mask = [1] * len(tokens)

            else:
                raise ValueError(
                    "Dataset example must contain either 'messages' or 'text'."
                )

            if count_examples:
                self.current_example += 1
            return tokens, assistant_mask

        # ---------------------------------------------------------
        # 2. Tight-packing iterator
        # ---------------------------------------------------------
        def packed_examples():

            token_buffer = []
            mask_buffer = []

            for example in DATASET:
                tokens, assistant_mask = encode_example(example)

                token_buffer.extend(tokens)
                mask_buffer.extend(assistant_mask)

                # We need max_length + 1 because of causal shifting:
                #
                # tokens:
                #   [t0 t1 t2 ... tN]
                #
                # input:
                #   [t0 t1 t2 ... tN-1]
                #
                # label:
                #   [t1 t2 t3 ... tN]
                #
                while len(token_buffer) >= max_length + 1:
                    chunk_tokens = token_buffer[: max_length + 1]
                    chunk_mask = mask_buffer[: max_length + 1]

                    del token_buffer[: max_length + 1]
                    del mask_buffer[: max_length + 1]

                    # Causal LM shift
                    input_ids = chunk_tokens[:-1]
                    target_ids = chunk_tokens[1:]

                    # Shift the assistant mask as well.
                    #
                    # mask[i] corresponds to token[i].
                    # label[i] corresponds to token[i + 1].
                    target_mask = chunk_mask[1:]

                    labels = [
                        token if mask else self.config.label_idx
                        for token, mask in zip(
                            target_ids,
                            target_mask,
                        )
                    ]

                    yield {
                        "data": input_ids,
                        "labels": labels,
                    }

        # ---------------------------------------------------------
        # 3. Collate fixed-length packed samples: Normal List --> torch tensor
        # ---------------------------------------------------------
        def collate_fn(batch):

            data = torch.tensor(
                [item["data"] for item in batch],
                dtype=torch.long,
            )

            labels = torch.tensor(
                [item["labels"] for item in batch],
                dtype=torch.long,
            )

            return {
                "data": data,
                "labels": labels,
            }

        # ---------------------------------------------------------
        # 4. DataLoader
        # ---------------------------------------------------------
        # Both branches below are functionally identical (as in the
        # original code) - `DATASET` is already this rank's shard, so we
        # just wrap it in a plain IterableDataset either way. No
        # DistributedSampler is needed since sharding already happened at
        # the `datasets.Dataset.shard(...)` level in __init__.
        if streaming:

            class PackedDataset(TorchIterableDataset):
                def __iter__(self):
                    yield from packed_examples()

            loader = DataLoader(
                PackedDataset(),
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                pin_memory=(self.device.type == "cuda"),
                num_workers=0,
            )

        else:

            class PackedDataset_NonStreaming(TorchIterableDataset):
                def __iter__(self):
                    yield from packed_examples()

            loader = DataLoader(
                PackedDataset_NonStreaming(),
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                pin_memory=(self.device.type == "cuda"),
                num_workers=0,
            )

        return loader
    
    
class _nullcontext:
    """Tiny stand-in for contextlib.nullcontext (kept local so this file
    has no extra stdlib import surprises) - used as the "no-op" branch of
    the `model.no_sync()` / plain-context choice during grad accumulation.
    """

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False