import csv
import math
import subprocess
import time
from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F

from torch.amp import autocast
from tqdm import tqdm

from Model.loss import get_cross_entropy_loss
from tqdm.contrib.logging import logging_redirect_tqdm


class Trainer(ABC):
    def get_entropy_and_mean_token_accuracy(self, logits, labels, label_idx):
        flat_logits = logits.view(-1, logits.size(-1))
        flat_labels = labels.view(-1)
        mask = flat_labels != label_idx
        
        valid_logits = flat_logits[mask]
        valid_labels = flat_labels[mask]
        
        log_probs = F.log_softmax(valid_logits, dim=-1)
        probs = log_probs.exp()
        
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        
        predictions = valid_logits.argmax(dim=-1)
        
        mean_token_accuracy = (
            predictions == valid_labels
        ).float().mean()
        
        return entropy.item(), mean_token_accuracy.item()

    
    @abstractmethod
    def train(self) -> Any:
        pass

    def get_learning_rate(self, pipeline):
        match pipeline:
            case "PT":
                return 3e-4

            case _:
                return 1e-4

    def load_dataset_from_local(self, datapath: Dict[str, Any]):
        data = {}

        for split, files in datapath.items():
            data[split] = {}

            for name, path in files.items():
                data[split][name] = np.memmap(path, dtype=np.uint16, mode="r")

        return data

    # -------------------------------------------------
    # MEMORY MANAGEMENT: GPU TEMPERATURE, COOLDOWN, ALLOCATIONS & LIMIT INFO
    # -------------------------------------------------
    def get_gpu_temperature(self):
        """Queries the NVIDIA GPU temperature using nvidia-smi."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            return int(result.stdout.strip())
        except Exception:
            return None

    def check_and_cooldown_gpu(self, max_temp=75, cooldown_temp=60):
        """Checks the GPU temperature and pauses training if it exceeds the limit."""
        temp = self.get_gpu_temperature()
        if temp is not None and temp >= max_temp:
            print(
                f"\n[Thermal Guard] GPU Temperature reached {temp}°C (limit: {max_temp}°C). Pausing training to cooldown..."
            )
            while temp is not None and temp > cooldown_temp:
                time.sleep(10)
                temp = self.get_gpu_temperature()
                print(
                    f"  [Thermal Guard] Cooling down -- Current temp: {temp}°C / Target: {cooldown_temp}°C"
                )
            print(f"\n[Thermal Guard] GPU cooled down to {temp}°C. Resuming training.")

    def get_allocated_memory(self, device: str):
        if device == "cuda":
            return torch.cuda.memory_allocated(device=torch.cuda.current_device()) / (
                1024 * 1024
            )

    def check_vram_limit(self, vram_limit, device):
        """Warns or clears cache if allocated memory exceeds the limit."""
        if device.type == "cuda":
            allocated = self.get_allocated_memory(device.type)

            if allocated > vram_limit:
                print(
                    f"\n[Memory Guard] Allocated [{allocated:.1f}]MB memory exceeds limit ({vram_limit} MB). Emptying cache..."
                )
                torch.cuda.empty_cache()

    def enforce_gpu_limit(self, vram_limits):
        curr_dev = torch.cuda.current_device()
        total_mem = torch.cuda.get_device_properties(curr_dev).total_memory / (
            1024 * 1024
        )
        if vram_limits < total_mem:
            fraction = vram_limits / total_mem
            torch.cuda.set_per_process_memory_fraction(fraction, device=curr_dev)
            print(
                f"[Memory Guard] Configured CUDA memory fraction to {fraction:.4f} (limits VRAM to ~{vram_limits} MB of total {total_mem:.1f} MB)"
            )
        else:
            print(
                f"[Memory Guard] Requested VRAM limit ({vram_limits} MB) exceeds total device VRAM ({total_mem:.1f} MB). Limit not set."
            )

    # -------------------------------------------------
    # LOGGING: METRICS, MODEL INTERNALS
    # -------------------------------------------------
    def log_metrics(self, csv_path, values):
        file = open(csv_path, "a", newline="")

        writer = csv.writer(file)
        writer.writerow(values)

        file.flush()

    @torch.no_grad()
    def evaluate_loss(
        self,
        model,
        reference_model,
        reference_checkpoint_dir,
        data,
        data_len,
        batch_size,
        max_seq_len,
        device,
        pipeline: str,
        eval_steps=None,
        desc = "Validation"
    ):
        model.eval()

        total_loss = 0.0

        skipped_batch = 0
        if not eval_steps:
            eval_steps = max(1, data_len // (batch_size * max_seq_len)) 
            
        for step in tqdm(range(eval_steps), desc=desc, leave=False):
            batch = self.get_batch(
                step,
                data,
                data_len,
                batch_size,
                max_seq_len,
                device,
                pipeline,
                rand=False,
            )

            with autocast(device_type=device.type, dtype=torch.float16):
                logits = model(batch["inputs"])
                loss = get_cross_entropy_loss(logits, batch["labels"])

            if torch.isnan(loss):
                skipped_batch += 1
                continue
            total_loss += loss.item()

        torch.cuda.empty_cache()

        model.train()

        return total_loss / (eval_steps - skipped_batch)