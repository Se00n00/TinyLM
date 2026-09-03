import csv
import math
import os
import time
import json
from ast import Pass
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from datasets import NamedSplit
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from datasets.arrow_dataset import Dataset
from datasets.iterable_dataset import IterableDataset as HFIterableDataset
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset as TorchIterableDataset
from tqdm import tqdm
from typing import Any
from trainer.base import Trainer
from trainer.sfttrainer.sftconfig import SFTConfig
from trainer.util import get_lr


class SFTTrainer(Trainer):
    def __init__(
        self,
        training_name,
        training_config,
        pipeline,
        dataset_name,
        model,
        tokenizer,
        ds: Dataset | HFIterableDataset,
        config: SFTConfig,
        pre_resumption_pipeline:Any =None,
    ):
        super().__init__(
           training_name,
            training_config,
            pipeline,
            dataset_name,
            model,
            tokenizer, 
            config, 
            pre_resumption_pipeline
        )

        if isinstance(ds, Dataset):
            Data = ds.train_test_split(test_size=config.test_train_ratio)
            train_data = Data["train"]
            test_data = Data["test"]
            self.test_samples = test_data.num_rows
            self.train_samples = train_data.num_rows
            self.iterable_train_data = False

            if self.is_ddp:
                train_data = train_data.shard(
                    num_shards=self.world_size, index=self.rank, contiguous=True
                )

        else:
            total_test = int(config.total_samples * config.test_train_ratio)
            total_train = config.total_samples - total_test

            # Full, unsharded eval stream - only rank 0 ever iterates this.
            test_ = ds.skip(total_train)
            test_items = list(test_)
            self.test_samples = len(test_items)

            if len(test_items) > 0:
                test_data = Dataset.from_list(test_items)
            else:
                # No exception here either way — `Dataset.from_list([])` still hits
                # SchemaInferenceError with zero examples unless you hand it features.
                test_data = Dataset.from_list([], features=ds.features)

            self.test_samples = total_test

            train_stream = ds.take(total_train)

            if self.is_ddp:
                per_rank = total_train // self.world_size
                start = self.rank * per_rank
                train_data = train_stream.skip(start).take(per_rank)
                self.train_samples = total_train
            else:
                train_data = train_stream
                self.train_samples = total_train

            self.iterable_train_data = True

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

        self.scaler = torch.amp.GradScaler(
            self.device.type, enabled=(self.config.ptdtype == torch.float16)
        )

        

        self._barrier()

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

    def _all_ranks_out_of_data(self, local_out_of_data: bool) -> bool:
        """
        All-reduces a per-rank 'ran out of data' flag (MIN) so every rank
        agrees on whether the WHOLE distributed epoch is over, not just
        this rank's own shard. Must be called by every rank at the same
        point in the loop, every micro-step - this is itself the
        synchronization point that keeps ranks from diverging on
        collective call count.
        """
        if not self.is_ddp:
            return local_out_of_data

        t = torch.tensor(0.0 if local_out_of_data else 1.0, device=self.device)
        dist.all_reduce(
            t, op=dist.ReduceOp.MIN
        )  # stays 1 only if EVERY rank still has data
        return t.item() == 0.0

    def _any_rank_bad_loss(self, local_bad: bool) -> bool:
        """
        All-reduces a per-rank 'this rank's loss was non-finite' flag (MAX)
        so every rank agrees on whether to abandon this micro-batch step,
        instead of one rank silently skipping a collective call that the
        others still expect.
        """
        if not self.is_ddp:
            return local_bad

        t = torch.tensor(1.0 if local_bad else 0.0, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return t.item() == 1.0

    def _any_rank_oom(self, local_oom: bool) -> bool:
        """
        All-reduces a per-rank 'this rank just OOM'd' flag (MAX) so every
        rank makes the same batch-size-shrink/bail decision on the same
        iteration, instead of one rank recovering locally while the
        others are still waiting inside a training collective it never
        rejoins.
        """
        if not self.is_ddp:
            return local_oom

        t = torch.tensor(1.0 if local_oom else 0.0, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return t.item() == 1.0

    def _broadcast_int(self, value: int, src: int = 0) -> int:
        """Broadcasts an int from `src` so every rank branches on an
        identical value instead of its own possibly-drifted local counter."""
        if not self.is_ddp:
            return value
        t = torch.tensor(int(value), dtype=torch.int64, device=self.device)
        dist.broadcast(t, src=src)
        return int(t.item())

    def _barrier(self):
        if self.is_ddp:
            dist.barrier()

    # -----------------------------------------------------------------
    # LAUNCH HELPERS (mp.spawn, for when you don't want to use torchrun)
    # -----------------------------------------------------------------
    @staticmethod
    def launch(
        world_size: int, master_addr="localhost", master_port="29500", **trainer_kwargs
    ):
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

        del trainer
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Run multiprocessing's own atexit cleanup explicitly - this is what
        # unlinks the semaphores mp.spawn's process-sync primitives created.
        # os._exit() below skips atexit entirely, which is what caused the
        # "leaked semaphore objects" warning once we started using it.
        import multiprocessing.util
        multiprocessing.util._exit_function()

        os._exit(0)

    

    def train(self):
        super().train()
        try:
            self._train_loop()
        finally:
            self._cleanup_ddp()

    def _train_loop(self):
        BATCH_SIZE = self.config.batch_size
        GRAD_ACCUM_STEPS = self.config.grad_accum_steps
        EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRAD_ACCUM_STEPS

        num_params = sum(p.numel() for p in self.model.parameters())

        step = 0
        loss_accum = float("inf")
        lr = 0
        grad_norm = torch.tensor(0.0)
        entropy, mean_token_accuracy = 0, 0
        self.best_val_loss = float("inf")

        if self.config.resume:
            resume_path = os.path.join(
                self.config.checkpoint_dir,
                f"{self.training_name}/{self.pre_resumption_pipeline}/model.pt",
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

                # step = checkpoint.get("step", 0) + 1
                # if self.config.resum_same_dataset == False:
                #     step = 0
                # else:
                #     # Continuing on the same dataset: restore how far
                #     # into it we were (this drives both the LR schedule
                #     # below and, more importantly, is what your training
                #     # script should have used to `.skip(...)` the raw
                #     # dataset via `SFTTrainer.peek_checkpoint(...)`
                #     # BEFORE it was ever passed into this trainer -
                #     # otherwise you'll be training over the same
                #     # examples again from the start of the stream.
                #     self.current_example = checkpoint.get(
                #         "current_example", self.current_example
                #     )

                self.current_example = (
                    self.current_example // self.world_size
                )  # Current Example for Each

                self.best_val_loss = checkpoint.get("val_loss", float("inf"))
                print(f"\nRESUMING CHECKPOINT: {self.pre_resumption_pipeline}/model.pt | VAL LOSS: {self.best_val_loss} | CURRENT EXAMPLE: {self.current_example}")
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
        t_start = time.time()

        total_iterations = (
            self.train_samples // self.world_size if self.is_ddp else self.train_samples
        )
        if self.is_main_process:
            print(
                "\n-------------------------------------------------------------------------"
            )
            print(
                f"| DEVICE: {self.device.type} | PARAMETERS:{num_params / (1024 * 1024)}| "
                f"BATCH SIZE:{BATCH_SIZE} | WORLD SIZE: {self.world_size} |"
            )
            print(
                "-------------------------------------------------------------------------\n"
            )
        self._barrier()

        train_bar = None

        try:
            # NO WARMUP REQUIED FOR CONTINUED TRAINING
            # if self.config.resum_same_dataset == False and self.config.resume == True:
            #     self.current_example = int(
            #         total_iterations * self.config.warmup_steps_ratio
            #     )

            run_meta = {
                "vocab_size": len(self.tokenizer),
                "batch_size": self.config.batch_size,
                "max_seq_len": self.config.max_length,
                "learning_rate": self.config.learning_rate,
            }

            # ---------------------------------------------
            # TRAIN LOOP
            # ---------------------------------------------

            for epoch in range(self.config.epochs):
                self._barrier()
                train_bar = tqdm(
                    total=total_iterations,
                    desc="Training",
                    unit="ex",
                    dynamic_ncols=True,
                    disable=not self.is_main_process,
                )
                
                Batches = iter(
                    self.get_batch(self.train_data, streaming=self.iterable_train_data)
                )
                EPOCH_COMPLETED = False
                prev_log_example = -1
                prev_eval_example = -1
                while not EPOCH_COMPLETED:  # <-- Depends upon total_examples
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
                        self.current_example,
                        total_iterations,
                        self.config.learning_rate,
                        int(total_iterations * self.config.warmup_steps_ratio)
                        if self.config.warmup_steps_ratio > 0
                        else 2000,
                        self.config.min_lr_ratio,
                    )
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = lr
    
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
    
                            local_out_of_data = batch is None
                            all_out_of_data = self._all_ranks_out_of_data(local_out_of_data)
    
                            if all_out_of_data:
                                EPOCH_COMPLETED = True
                                break
    
                            is_last_micro_step = (micro_step == GRAD_ACCUM_STEPS - 1)
                            sync_context = (
                                self.model.no_sync()
                                if self.is_ddp and not is_last_micro_step
                                else _nullcontext()
                            )
    
                            # ---------------------------------------------------
                            # Every rank must call EXACTLY the same sequence of
                            # collectives this micro-step regardless of which of
                            # the three cases below it hits (out-of-data locally,
                            # empty/all-masked batch, or a normal batch) - or the
                            # ranks permanently desync their NCCL call count and
                            # hang 30 min later on some future collective. So we
                            # decide `local_bad_loss` uniformly first, always
                            # call `_any_rank_bad_loss` exactly once, and then
                            # always call exactly one backward() (real loss or
                            # zero dummy loss) under the same sync_context.
                            # ---------------------------------------------------
                            if local_out_of_data:
                                local_bad_loss = False
                                is_empty_batch = True
                            else:
                                with autocast(
                                    device_type=self.device.type, dtype=self.config.ptdtype
                                ):
                                    batch_x = batch["data"].to(self.device, non_blocking=True)
                                    batch_y = batch["labels"].to(self.device, non_blocking=True)
    
                                    valid_tokens = (batch_y != self.config.label_idx).sum()
    
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
    
                                is_empty_batch = valid_tokens.item() == 0
                                local_bad_loss = (
                                    False if is_empty_batch else not torch.isfinite(raw_loss)
                                )
    
                            any_bad_loss = self._any_rank_bad_loss(local_bad_loss)
    
                            if any_bad_loss:
                                if local_bad_loss and self.is_main_process:
                                    print("Non-finite loss encountered.")
                                self.optimizer.zero_grad(set_to_none=True)
                                step_completed = False
                                break
    
                            if local_out_of_data:
                                zero_loss = sum(
                                    (p.sum() for p in self.model.parameters()),
                                    start=torch.zeros((), device=self.device),
                                ) * 0.0
                                with sync_context:
                                    zero_loss.backward()
                                continue
    
                            if is_empty_batch:
                                zero_loss = (logits.sum() * 0.0) / GRAD_ACCUM_STEPS
                                with sync_context:
                                    zero_loss.backward()
                                consumed += 1
                                continue
    
                            valid_micro_steps += 1
                            loss_accum += raw_loss.item()
                            entropy_accum += batch_entropy
                            accuracy_accum += batch_accuracy
    
                            loss = raw_loss / GRAD_ACCUM_STEPS
    
                            with sync_context:
                                if self.config.ptdtype == torch.float16:
                                    self.scaler.scale(loss).backward()
                                else:
                                    loss.backward()
    
                            consumed += 1
    
                        if valid_micro_steps == 0:
                            self.optimizer.zero_grad(set_to_none=True)
                            step_completed = False
                            if EPOCH_COMPLETED:
                                break
                            continue
    
                        loss_accum /= GRAD_ACCUM_STEPS
                        entropy = entropy_accum / GRAD_ACCUM_STEPS
                        mean_token_accuracy = accuracy_accum / GRAD_ACCUM_STEPS
    
                        if self.config.ptdtype == torch.float16:
                            self.scaler.unscale_(self.optimizer)
    
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
    
                    sync_example = self._broadcast_int(self.current_example, src=0)
    
                    if self.is_main_process:
                        postfix = {
                            "remaining": max(0, total_iterations - self.current_example)
                        }
                        if elapsed > 0:
                            postfix["Batch Throughput"] = f"{consumed / elapsed:.2f}"
                            postfix["Batchs"] = f"{consumed}"
                        train_bar.set_postfix(postfix)
    
                    # -------------------------------------------------------
                    # EVAL STEP/
                    # -------------------------------------------------------
                    if ( sync_example % self.config.eval_steps == 0 and sync_example > 0 and prev_eval_example != sync_example ) or (
                        EPOCH_COMPLETED == True
                    ):
                        self.model.eval()
                        val_loss = 0.0
                        val_entropy, val_mean_token_accuracy = 0.0, 0.0
                        prev_eval_example = sync_example
    
                        if self.is_main_process:
                            total_eval_loss = 0.0
                            total_valid_tokens = 0
                            eval_steps = 0
    
                            val_pbar = tqdm(desc="Initial Validation" if sync_example == 0 else "Validation")
                            for batch in self.get_batch(
                                self.test_data, streaming=False, count_examples=False
                            ):
                                batch_x = batch["data"].to(self.device, non_blocking=True)
                                batch_y = batch["labels"].to(self.device, non_blocking=True)
    
                                valid_tokens = (batch_y != self.config.label_idx).sum()
    
                                if valid_tokens.item() == 0:
                                    val_pbar.update(1)
                                    continue
    
                                with autocast(
                                    device_type=self.device.type, dtype=torch.float16
                                ):
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
    
                                total_eval_loss += loss.item()
                                total_valid_tokens += 1
                                val_entropy += batch_entropy
                                val_mean_token_accuracy += batch_mean_token_accuracy
                                eval_steps += 1
                                val_pbar.update(1)
                            val_pbar.close()
    
                            if total_valid_tokens > 0:
                                val_loss = total_eval_loss / total_valid_tokens
                            else:
                                val_loss = float("nan")
    
                            if eval_steps > 0:
                                val_entropy /= eval_steps
                                val_mean_token_accuracy /= eval_steps
                            else:
                                val_entropy = float("nan")
                                val_mean_token_accuracy = float("nan")
    
                        # print(f"VALIDATION: {self.local_rank}")
                        # Make sure no rank races ahead into more training
                        self._barrier()
    
                        global_current_example = self._reduce_sum(self.current_example)
    
                        val_ppl = math.exp(val_loss) if val_loss < 1000 else float("inf")
                        train_ppl = (
                            math.exp(loss_accum) if loss_accum < 1000 else float("inf")
                        )
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
                                f"Current Example {sync_example} | "
                                f"Train Loss: {loss_accum:.4f} (PPL: {train_ppl:.2f}) | "
                                f"Val Loss: {val_loss:.4f} (PPL: {val_ppl:.2f}) | "
                                f"Grad Norm: {grad_norm:.2f} | "
                                f"Step Time: {elapsed * 1000:.1f}ms | "
                                f"VRAM: {allocated_mb:.1f}MB" + temp_str
                            )
    
                            if val_loss < self.best_val_loss:
                                self.best_val_loss = val_loss
                                checkpoint_path = os.path.join(
                                    self.config.checkpoint_dir,
                                    f"{self.training_name}/{self.pipeline}/best.pt",
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
                                f"{self.config.checkpoint_dir}/{self.training_name}/{self.pipeline}/logs_validation.csv",
                                [
                                    self.training_config['global_current_example'] + (self.config.logging_steps * self.world_size),
                                    int(global_current_example),
                                    val_loss,
                                    val_ppl,
                                    val_entropy,
                                    val_mean_token_accuracy,
                                ],
                            )
                        # print(f"VALIDATION --> SAVING: {self.local_rank}")
                        # Make sure no rank races ahead into more training
                        # steps while rank 0 is still writing the checkpoint.
                        self._barrier()
    
                    # TRAINING LOGGING: STEP, NUM_TOKENS, LOSS, PERPLEXITY, ENTROPY, MEAN_TOKEN_ACCURACY, LR, GRAD_NORM
                    # print(f"\nEXAMPLE: {self.current_example} | PREV_LOG_EXAMPLE: {prev_log_example} | LOG: {self.current_example % self.config.logging_steps == 0 and self.current_example != prev_log_example}\n")
                    if ( sync_example % self.config.logging_steps == 0 and sync_example != prev_log_example ) or (
                        EPOCH_COMPLETED == True
                    ):
                        prev_log_example = sync_example
    
                        global_current_example_log = self._reduce_sum(self.current_example)
    
                        if self.is_main_process:
                            train_ppl = (
                                math.exp(loss_accum) if loss_accum < 1000 else float("inf")
                            )
    
                            self.log_metrics(
                                f"{self.config.checkpoint_dir}/{self.training_name}/{self.pipeline}/logs_train.csv",
                                [
                                    self.training_config['global_current_example'] + (self.config.logging_steps * self.world_size),
                                    sync_example * self.world_size,
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
                                f"{self.training_name}/{self.pipeline}/model.pt",
                            )
                            torch.save(
                                {
                                    "step": step,
                                    "val_loss": self.best_val_loss,
                                    "model_state_dict": self.raw_model.state_dict(),
                                    "optimizer_state_dict": self.optimizer.state_dict(),
                                    "run_meta": run_meta,
                                },
                                checkpoint_path,
                            )
    
                            config_data = self.training_config
                            target_idx = next(
                                idx
                                for idx, d in enumerate(
                                    self.training_config["pipeline"][self.pipeline]
                                )
                                if d["dataset"] == self.dataset_name
                            )
    
                            config_data["pipeline"][self.pipeline][target_idx][
                                "trained"
                            ] = global_current_example_log
                            config_data["pipeline"][self.pipeline][target_idx][
                                "completed"
                            ] = bool(EPOCH_COMPLETED)
    
                            # print(f"\n\nCURRENT EXAMPLE: {config_data['global_current_example']} | INCREMENT: {(self.config.logging_steps * self.world_size) if sync_example > 0 else 0} | REAL GLOBAL EXAMPLE: {global_current_example_log} | EXPECTED {sync_example * self.world_size}\n\n")
                            config_data['global_current_example'] += (self.config.logging_steps * self.world_size) if sync_example > 0 else 0
                            config_data['current_pipeline'] = self.pipeline
    
                            training_path = os.path.join(
                                f"{self.config.checkpoint_dir}/{self.training_name}",
                                "training.yaml",
                            )
                            with open(training_path, "w") as file:
                                yaml.safe_dump(
                                    config_data,
                                    file,
                                    sort_keys=False,
                                    default_flow_style=False,
                                )
    
                        # print(f"VALIDATION --> SAVING --> CONFIG & NORMAL SAVE: {self.local_rank}")
                        self._barrier()
    
                    # -------------------------------------------------------
                    # FINAL CHECKPOINT — ALWAYS SAVE WHEN EPOCH COMPLETES
                    # -------------------------------------------------------
                    if EPOCH_COMPLETED:
                        self._barrier()
    
                        if self.is_main_process:
                            final_checkpoint_path = os.path.join(
                                self.config.checkpoint_dir,
                                f"{self.training_name}/{self.pipeline}/model.pt",
                            )
    
                            final_tmp_path = final_checkpoint_path + ".tmp"
    
                            checkpoint = {
                                "step": step,
                                "val_loss": self.best_val_loss,
                                "current_example": int(
                                    self._reduce_sum(self.current_example)
                                ),
                                "model_state_dict": self.raw_model.state_dict(),
                                "optimizer_state_dict": self.optimizer.state_dict(),
                                "run_meta": run_meta,
                            }
    
                            # Write temporary file first
                            torch.save(checkpoint, final_tmp_path)
    
                            # Atomic replacement
                            os.replace(final_tmp_path, final_checkpoint_path)
    
                            print(
                                f"\n[FINAL CHECKPOINT] Saved successfully:\n"
                                f"{final_checkpoint_path}\n"
                                f"step={step}\n"
                                f"current_example={checkpoint['current_example']}\n"
                            )
    
                        self._barrier()
    
                    step += 1
    
                if self.is_main_process:
                    train_bar.close()

            # print(f"TIMES INCREMENTED: {incremented} | INCREMENT AMOUNT: {self.config.logging_steps * self.world_size}")
        except RuntimeError as e:
            tqdm.write(str(e))
            err_msg = str(e).lower()
            local_is_oom = (
                "out of memory" in err_msg
                or "memory limit" in err_msg
                or "allowed memory" in err_msg
            )

            # Every rank must agree whether this was recoverable OOM,
            # otherwise one rank could shrink+retry while others fall
            # through to cleanup and hang the survivors on the next
            # collective call.
            any_oom = self._any_rank_oom(local_is_oom)

            if not any_oom:
                # Non-OOM error on at least one rank: everyone raises
                # together rather than leaving healthy ranks stuck in a
                # collective the failing rank never rejoins.
                train_bar.close()
                raise e

            if self.is_main_process:
                print(
                    f"\n[Memory Guard] CUDA OOM or limit exceeded at step {step} "
                    f"with batch_size={BATCH_SIZE}, grad_accum_steps={GRAD_ACCUM_STEPS}."
                )
            self.optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            self._barrier()  # let every rank finish cleanup before retrying

            if BATCH_SIZE > 1:
                old_bs = BATCH_SIZE
                BATCH_SIZE = max(1, BATCH_SIZE // 2)
                GRAD_ACCUM_STEPS = EFFECTIVE_BATCH_SIZE // BATCH_SIZE
                if self.is_main_process:
                    print(
                        f"  [Memory Guard] Halving micro-batch size: {old_bs} -> {BATCH_SIZE}. "
                        f"Increasing grad_accum_steps to {GRAD_ACCUM_STEPS}."
                    )
            elif not getattr(self.raw_model, "gradient_checkpointing", False):
                if self.is_main_process:
                    print(
                        "  [Memory Guard] Micro-batch size is already 1. "
                        "Enabling gradient checkpointing to save memory..."
                    )
                self.raw_model.gradient_checkpointing = True
                BATCH_SIZE = self.config.batch_size
                GRAD_ACCUM_STEPS = self.config.grad_accum_steps
            else:
                if self.is_main_process:
                    print(
                        "  [Memory Guard] Out of memory even with micro-batch size 1 "
                        "and gradient checkpointing active."
                    )
                train_bar.close()
                raise e

            # Re-enter the training loop with the adjusted settings instead
            # of falling through — this is the part the original code never
            # did, so "recovery" never actually resumed training.
            return self._train_loop()

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
                    if tools:
                        if isinstance(tools, str):
                            try:
                                tools = json.loads(tools)
                            except json.JSONDecodeError:
                                try:
                                    import ast
                                    tools = ast.literal_eval(tools)
                                except (ValueError, SyntaxError):
                                    # Invalid tool definition → skip example
                                    return None, None

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
                
                if tokens is None or assistant_mask is None:
                    continue

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
