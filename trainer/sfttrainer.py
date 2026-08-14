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

from ast import Pass
import time
from dataclasses import dataclass
import os
import csv
from pathlib import Path
import datasets
import torch
import math
from datasets.arrow_dataset import Dataset
from datasets.iterable_dataset import IterableDataset as HFIterableDataset
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset as TorchIterableDataset
from tqdm import tqdm
from torch.amp import autocast
import torch.nn.functional as F

from .base import Trainer
from .util import get_lr

@dataclass
class SFTConfig:
    # MISC
    total_samples:int
    test_train_ratio: float = 0.1
    min_lr_ratio = 0.8
    max_test_rows = 10000
    label_idx = -100

    # LEARNING PARAMETERS 
    batch_size: int = 2
    grad_accum_steps: int = 16
    warmup_steps_ratio:float = 0.10 # 10 % of total steps
    checkpoint_dir:str = "checkpoints"
    resume:str | None = None
    resum_same_dataset:bool = False
    learning_rate: float = 2e-5
    
    # LOGGING & EVALUATION
    logging_steps: int = 10
    eval_steps:int = 200
    
    # MODEL PARAMTERS
    max_length: int = 512
    ptdtype = torch.float16
    
    # MEMORY MANAGEMENT
    vram_limit_mb:int = 4000
    max_temp:int = 75
    cooldown_temp:int = 60
    
    # TRAINING
    gradient_checkpointing: bool = True
    weight_decay:float = 0.1
    
    # Trainer Variable [should be put into SFTTrainer init]
    current_example = 0 

class SFTTrainer(Trainer):
    def __init__(
        self,
        training_name,
        model,
        tokenizer,
        ds: Dataset | HFIterableDataset,
        config: SFTConfig
    ):
        self.training_name = training_name
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        

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
        self.train_data = train_data
        self.test_data = test_data

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            weight_decay=self.config.weight_decay,
            lr=self.config.learning_rate,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=(self.config.ptdtype == torch.float16))
        
        # Initiallize .csv files for Logging
        self.init_csv(
            f"{config.checkpoint_dir}/{training_name}/logs_train.csv",
            ["step", "num_examples", "loss", "perplexity", "entropy", "mean_token_accuracy", "learning_rate", "GNorm"],
        )
        self.init_csv(
            f"{config.checkpoint_dir}/{training_name}/logs_validation.csv",
            ["step", "num_examples", "loss", "perplexity", "entropy", "mean_token_accuracy"],
        )

    def init_csv(self, csv_path, cols):
        if not Path(csv_path).exists():
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(cols)
                
    def train(self):
        super().train()
        self.best_val_loss = float("inf")
        self.model.to(self.device)
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
                checkpoint = torch.load(
                    resume_path, map_location=self.device, weights_only=False
                )
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)

                step = checkpoint.get("step", 0) + 1
                if self.config.resum_same_dataset == False:
                    step = 0
                    
                self.best_val_loss = checkpoint.get("val_loss", float("inf"))
        
            else:
                print(
                    f"Warning: Checkpoint path '{resume_path}' not found. Starting training from scratch (step 0)."
                )
                
        consumed = 0
        t_initial = time.time()
        
        train_bar = tqdm(desc="Training", dynamic_ncols=True)
            
        try:
            total_iterations = self.config.total_samples
            
            if self.config.resum_same_dataset == False and self.config.resume == True:
                self.config.current_example = int(total_iterations * self.config.warmup_steps_ratio)
                
            
            run_meta = {
                "vocab_size": len(self.tokenizer),
                "batch_size": self.config.batch_size,
                "max_seq_len": self.config.max_length,
                "learning_rate": self.config.learning_rate,
            }
            print(
                "\n--------------------------------------------------------------------------"
            )
            print(
                f"DEVICE: {self.device.type} | PARAMETERS:{num_params / (1024 * 1024)}| BATCH SIZE:{BATCH_SIZE}"
            )
            print(
                "----------------------------------------------------------------------------\n"
            )
            # ---------------------------------------------
            # TRAIN LOOP
            # ---------------------------------------------
            
            Batches = iter(self.get_batch(self.train_data, streaming=self.iterable_train_data))
            EPOCH_COMPLETED = False
            while not EPOCH_COMPLETED:
                self.check_and_cooldown_gpu(self.config.max_temp, self.config.cooldown_temp)
                lr = get_lr(
                    self.config.current_example, total_iterations, self.config.learning_rate, int(total_iterations * self.config.warmup_steps_ratio), self.config.min_lr_ratio
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

                    self.check_vram_limit(self.config.vram_limit_mb, self.device)

                    for micro_step in range(GRAD_ACCUM_STEPS):
                        
                        global_micro_step = step * GRAD_ACCUM_STEPS + micro_step
                        
                        batch = next(Batches, None)
                        if batch is None:
                            EPOCH_COMPLETED = True
                            break
                            
                        with autocast(device_type=self.device.type, dtype=self.config.ptdtype):
                            batch_x = batch["data"].to(self.device, non_blocking=True)
                            batch_y = batch["labels"].to(self.device, non_blocking=True)
                            
                            logits = self.model(batch_x)
                            raw_loss = F.cross_entropy(
                                logits.view(-1, logits.size(-1)), batch_y.view(-1), ignore_index=self.config.label_idx
                            )
                            batch_entropy, batch_accuracy = self.get_entropy_and_mean_token_accuracy(logits, batch_y, self.config.label_idx)
                        
                        
                        if not torch.isfinite(raw_loss):
                            print("Non-finite loss encountered.")
                            self.optimizer.zero_grad(set_to_none=True)
                            step_completed = False
                            break
                    
                        loss_accum += raw_loss.item()
                        entropy_accum += batch_entropy
                        accuracy_accum += batch_accuracy

                        loss = raw_loss / GRAD_ACCUM_STEPS

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
            
                train_bar.update(1)
                elapsed = time.time() - t_start
    
                if elapsed > 0:
                    batches_per_second = consumed / elapsed
    
                    train_bar.set_postfix(
                        {
                            "batch/s": f"{batches_per_second:.2f}",
                        }
                    )
            
                # -------------------------------------------------------
                # EVAL STEP/
                # -------------------------------------------------------
                if step % self.config.eval_steps == 0 or (EPOCH_COMPLETED == True):
                    self.model.eval()
                    total_eval_loss = 0.0
                    val_entropy, val_mean_token_accuracy = 0,0
                    eval_steps = 0
                    
                    for batch in tqdm(self.get_batch(self.test_data, streaming=False), desc="Validation"):
                        batch_x = batch["data"].to(self.device, non_blocking=True)
                        batch_y = batch["labels"].to(self.device, non_blocking=True)
                        
                        with autocast(device_type=self.device.type, dtype=torch.float16):
                            logits = self.model(batch_x)
                            loss = F.cross_entropy(
                                logits.view(-1, logits.size(-1)), batch_y.view(-1), ignore_index=self.config.label_idx
                            )
                            batch_entropy,batch_mean_token_accuracy = self.get_entropy_and_mean_token_accuracy(logits, batch_y, self.config.label_idx)
                        
                            val_entropy += batch_entropy
                            val_mean_token_accuracy += batch_mean_token_accuracy
                        
                        total_eval_loss += loss.item()
                        eval_steps += 1
                        
                    
                    val_entropy = val_entropy / eval_steps
                    val_mean_token_accuracy = val_mean_token_accuracy / eval_steps
                    val_loss = total_eval_loss / eval_steps
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
                            f"{self.training_name}/model.pt",
                        )
                        torch.save(
                            {
                                "step": step,
                                "model_state_dict": self.model.state_dict(),
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
                        [step, self.config.current_example * BATCH_SIZE *self.config.max_length, val_loss, val_ppl, val_entropy, val_mean_token_accuracy ],
                    )
                
                # TRAINING LOGGING: STEP, NUM_TOKENS, LOSS, PERPLEXITY, ENTROPY, MEAN_TOKEN_ACCURACY, LR, GRAD_NORM
                if step % self.config.logging_steps == 0 or (EPOCH_COMPLETED == True):
                    train_ppl = math.exp(loss_accum) if loss_accum < 1000 else float("inf")
                    
                    self.log_metrics(
                        f"{self.config.checkpoint_dir}/{self.training_name}/logs_train.csv",
                        [step, self.config.current_example * BATCH_SIZE *self.config.max_length,  loss_accum, train_ppl, entropy, mean_token_accuracy, lr, grad_norm.item() ],
                    )
                    
                step += 1
            print("\nTRAINING COMPLETE !")
        except RuntimeError as e:
            tqdm.write(str(e))
            err_msg = str(e).lower()
            if (
                "out of memory" in err_msg
                or "memory limit" in err_msg
                or "allowed memory" in err_msg
            ):
                print(
                    f"\n[Memory Guard] CUDA OOM or limit exceeded at step {step} with batch_size={BATCH_SIZE}, grad_accum_steps={GRAD_ACCUM_STEPS}."
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
                    BATCH_SIZE = self.config.batch_size
                    GRAD_ACCUM_STEPS = self.config.grad_accum_steps
                else:
                    print(
                        "  [Memory Guard] Out of memory even with micro-batch size 1 and gradient checkpointing active."
                    )
                    
                    train_bar.close()
                    raise e
            else:
                train_bar.close()
                raise e
        
    
    def get_batch(self, DATASET, streaming=True):
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
                        msg
                        for msg in messages
                        if msg.get("role") != "available_tools"
                    ]
                    tokens = tokenizer.apply_chat_template(
                        chat_messages,
                        tokenize=True,
                        tools = tools,
                        add_generation_prompt=False,
                    )['input_ids']
    
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
                            tools = tools,
                            add_generation_prompt=True,
                        )
    
                        # Tokenize the assistant content itself.
                        content_tokens = tokenizer(
                            msg.get("content", ""),
                            add_special_tokens=False,
                        )["input_ids"]
    
                        start = len(prefix_tokens['input_ids'])
    
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
                    "Dataset example must contain either "
                    "'messages' or 'text'."
                )
            
            self.config.current_example += 1
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
    
                    chunk_tokens = token_buffer[:max_length + 1]
                    chunk_mask = mask_buffer[:max_length + 1]
    
                    del token_buffer[:max_length + 1]
                    del mask_buffer[:max_length + 1]
    
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