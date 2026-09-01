# TinyLM Trainer
```

    ░▀█▀░▀█▀░█▀█░█░█░█░░░█▄█░░░▀█▀░█▀▄░█▀█░▀█▀░█▀█░█▀▀░█▀▄
    ░░█░░░█░░█░█░░█░░█░░░█░█░░░░█░░█▀▄░█▀█░░█░░█░█░█▀▀░█▀▄
    ░░▀░░▀▀▀░▀░▀░░▀░░▀▀▀░▀░▀░░░░▀░░▀░▀░▀░▀░▀▀▀░▀░▀░▀▀▀░▀░▀
    
```
A CLI-driven, checkpoint-resumable, multi-pipeline SFT/PT trainer with optional
multi-GPU DDP, thermal/VRAM guards, and automatic OOM recovery.

This document describes what the code actually does. Where the old draft of
this file disagreed with the implementation (extra CLI flags, wrong module
paths, wrong resume syntax, etc.) it has been corrected below, and the
discrepancies are called out explicitly in [§12 Known Rough Edges](#12-known-rough-edges-read-before-you-debug)
so nobody loses an afternoon to them twice.

---

## 1. Overview

Training is organized as a sequence of **pipelines** , each pipeline containing one or more
**datasets**. `train.py` walks that list in order, skipping anything already
finished, and hands each dataset to `SFTTrainer` one at a time. Progress
(which dataset/pipeline you're on, how many rows you've consumed) is
persisted to a per-run `training.yaml`, so a killed job can be restarted with
`--resume` and it will pick up where it left off — including mid-dataset, via
row skipping on the underlying HuggingFace dataset/stream.

Core pieces:

| Component | File | Responsibility |
|---|---|---|
| `Train` | `train.py` | CLI entry point. Reads `config.yaml` + `training.yaml`, loads model/tokenizer, iterates pipelines/datasets, applies per-pipeline preprocessing, constructs `SFTConfig` + `SFTTrainer`, launches training. |
| `Trainer` (ABC) | `trainer/base.py` | Shared plumbing: DDP bootstrap, CSV logging setup, GPU temperature/VRAM guards, entropy/accuracy computation. `train()` is abstract. |
| `SFTTrainer` | `trainer/sfttrainer/sfttrainer.py` | Concrete trainer. Dataset sharding, packing/tokenization, the full train/eval loop, checkpointing, DDP collectives, OOM recovery, `mp.spawn`-based launch. |
| `SFTConfig` | `trainer/sfttrainer/sftconfig.py` | Dataclass of all per-run/per-dataset hyperparameters. |
| `get_lr` | `trainer/util.py` | LR schedule (warmup + decay), consumed each outer step. |

---

## 2. Quick Start

### Fresh run

1. Edit `config.yaml` (next to `train.py`) to declare your pipeline order and
   the datasets in each pipeline (see [§5](#5-configyaml-what-you-hand-edit)).
2. If you're adding a new pipeline name, add a matching case to
   `Train._change_template` in `train.py` (see [§8](#8-pipelines--preprocessing)) — an
   unmatched pipeline name silently falls through to the pretraining
   (`preprocess_text_generation`) branch, which is almost never what you want
   for anything but `PT`.
3. Run:

```bash
python train.py \
  --training_name my_run \
  --batch_size 1 \
  --grad_accum_steps 4 \
  --max_seq_len 4096 \
  --stream_dataset True \
  --warmup_steps_ratio 0.10 \
  --validation_dataset_limit 1000
```

This creates `checkpoints/my_run/training.yaml` from scratch and starts on
the first pipeline/dataset in `config.yaml`.

### Resuming

Just add `--resume`. **Do not** pass a checkpoint path — `--resume` is a bare
flag; the trainer figures out where it left off from
`checkpoints/<training_name>/training.yaml`:

```bash
python train.py \
  --training_name my_run \
  --batch_size 1 \
  --grad_accum_steps 4 \
  --max_seq_len 4096 \
  --stream_dataset True \
  --warmup_steps_ratio 0.10 \
  --validation_dataset_limit 1000 \
  --resume
```

Make sure `config.yaml`'s pipeline **names and order** still match what's
recorded in `training.yaml` — the resume path keys checkpoints by pipeline
name (`<checkpoint_dir>/<training_name>/<pipeline>/model.pt`), not by
position, but the per-dataset `trained`/`completed` bookkeeping in
`training.yaml` assumes the same dataset list you started with.

### Multi-GPU via `torchrun` (preferred)

```bash
torchrun --nproc_per_node=4 train.py \
  --training_name my_run \
  --distributed ddp \
  --backend nccl \
  --batch_size 2 \
  --grad_accum_steps 8 \
  --max_seq_len 4096
```

`torchrun` sets `RANK` / `LOCAL_RANK` / `WORLD_SIZE`, so `Trainer._setup_ddp`
just attaches to the process group `torchrun` already created.

### Multi-GPU via built-in `mp.spawn` launcher

If you run `python train.py --distributed ddp --world_size 4` **without**
`torchrun` (i.e. `RANK` is not in the environment), `train.py` detects that
and calls `SFTTrainer.launch(world_size, **trainer_kwargs)` itself, which
spawns `world_size` local processes with `torch.multiprocessing`. This only
supports single-node, multi-GPU — use `torchrun` for multi-node.

```bash
python train.py \
  --training_name my_run \
  --distributed ddp \
  --world_size 4 \
  --batch_size 2 \
  --grad_accum_steps 8
```

---

## 3. CLI Reference

All flags are defined in `Train._parse_arguments` (`train.py`).

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--training_name` | str | — | Run name; also the checkpoint/log subfolder. |
| `--model` | choice | `Alibi` | Only `Alibi` is currently wired up (`Train.train`'s `match` falls back to the same branch for anything else). |
| `--batch_size` | int | `1` | Micro-batch size **per GPU**. Overridable per-dataset in `config.yaml` (`batch_size:`). |
| `--grad_accum_steps` | int | `4` | Micro-batches accumulated per optimizer step. Effective batch = `batch_size × grad_accum_steps × world_size`. Overridable per-dataset. |
| `--warmup_steps_ratio` | float | `0.10` | Fraction of `total_iterations` spent ramping LR from 0. |
| `--weight_decay` | float | `0.1` | AdamW weight decay. |
| `--disable_amp` | flag | off | **Parsed but not consumed anywhere in `SFTConfig`/`SFTTrainer` as shown** — mixed precision is controlled by `SFTConfig.ptdtype`, which is hardcoded to `torch.float16` and not exposed via CLI. See [§12](#12-known-rough-edges-read-before-you-debug). |
| `--gradient_checkpointing` | flag | off | Seeds `config.gradient_checkpointing`; the OOM-recovery path also force-enables it on `raw_model` if batch size is already 1. |
| `--checkpoint_dir` | str | `checkpoints` | Root dir for `<training_name>/training.yaml` and per-pipeline checkpoints/logs. |
| `--resume` | flag | off | Resume from `training.yaml` + the last pipeline's `model.pt`. No path argument. |
| `--validation_dataset_limit` | int | `None` | Caps eval-set size; also used to compute `test_train_ratio = limit / total_samples` when set. |
| `--stream_dataset` | bool | `False` | ⚠️ Uses `type=bool` in argparse — **any non-empty string, including `False`, evaluates truthy.** Pass `--stream_dataset True` deliberately or omit the flag; don't try `--stream_dataset False`. |
| `--max_seq_len` | int | `512` | Packed sequence length (tokens). Overridable per-dataset (`max_seq_len:`). |
| `--eval_interval` | int | `2000` | Run validation every N *examples consumed* (compared against `sync_example`, not optimizer steps). |
| `--log_interval` | int | `20` | Write a training-log row / rolling checkpoint every N examples. |
| `--vram_limit_mb` | int | `30000` | Soft cap; `check_vram_limit` empties CUDA cache when exceeded. Not a hard `set_per_process_memory_fraction` (that's `enforce_gpu_limit`, which nothing currently calls). |
| `--max_temp` | int | `75` | GPU temp (°C) that triggers a cooldown pause. |
| `--cooldown_temp` | int | `60` | Target temp before resuming after a thermal pause. |
| `--distributed` | choice | `none` | `none` or `ddp`. |
| `--backend` | choice | `nccl` | `nccl` or `gloo`. |
| `--world_size` | int | `1` | Only consulted for the `mp.spawn` self-launch path; ignored under `torchrun`. |

Flags **not present** despite appearing in older examples: `--pipeline`,
`--learning_rate` at the top level (learning rate is per-dataset in
`config.yaml`, defaulting to `3e-4`), and a path argument to `--resume`.

---

## 4. Directory Layout Produced

```
<checkpoint_dir>/<training_name>/
├── training.yaml                  # auto-managed resume state (see §6)
└── <pipeline_name>/
    ├── model.pt                   # rolling checkpoint, updated every log_interval
    ├── best.pt                    # best-val-loss checkpoint, updated on eval
    ├── logs_train.csv
    └── logs_validation.csv
```

---

## 5. `config.yaml` (what you hand-edit)

Lives next to `train.py`. Shape expected by `Train.train`:

```yaml
pipeline:
  - PT
  - IFT

dataset:
  PT:
    - base: some-org/some-pretrain-corpus
      split: train
      text_column: text
      # optional:
      subset: null
      data_dir: null
      limit: null
      batch_size: 1
      grad_accum_steps: 4
      learning_rate: 3e-4
      max_seq_len: 4096

  IFT:
    - base: some-org/some-chat-dataset
      split: train
      # text_column not needed for IFT — it uses the `messages` schema
```

Per-dataset keys read in `train.py`:

- `base` (required) — passed to `load_dataset_builder`/`load_dataset` as `path`.
- `split` (required).
- `subset` → forwarded as `name=`.
- `data_dir` → forwarded as `data_dir=`.
- `limit` → caps total examples used from this dataset (`ds.take(limit)`).
- `text_column` → required by the `PT`/default preprocessing branch.
- `batch_size`, `grad_accum_steps`, `learning_rate`, `max_seq_len` → override the corresponding CLI default for this dataset only.

If `builder.info.splits` doesn't know the split's row count, `total_samples`
falls back to `10_000_000_000` — meaning `test_train_ratio` math and the
progress bar total will be nonsense for datasets HF can't statically size.
Set `limit` explicitly for those.

---

## 6. `training.yaml` (auto-managed — don't hand-edit under `--resume`)

Created by `Train._initiallize_training_details` the first time a
`training_name` is used:

```yaml
current_pipeline: PT
global_current_example: 0
pipeline:
  PT:
    - dataset: some-org/some-pretrain-corpus
      trained: 0
      completed: false
  IFT:
    - dataset: some-org/some-chat-dataset
      trained: 0
      completed: false
```

Updated by `SFTTrainer._train_loop` every `log_interval` examples:

- `pipeline.<name>[i].trained` — cumulative examples trained on this dataset.
- `pipeline.<name>[i].completed` — set once the dataset's stream/shards are exhausted.
- `global_current_example` — incremented by `log_interval * world_size` each log tick (an approximation of total examples seen across the whole run, used only for the log CSV's step column).
- `current_pipeline` — the pipeline `train.py` was on when it started this dataset; on `--resume`, this is what tells `SFTTrainer` which pipeline's `model.pt` to load (`pre_resumption_pipeline`).

`train.py` treats a dataset as already done and skips it if
`training_samples <= trained` **or** `completed` is true — so if you shrink a
dataset's `limit` between runs, a partially-trained-but-now-"complete" entry
can be skipped even though it hasn't seen the new, smaller dataset — check
`trained`/`completed` manually if you change `limit` mid-run.

---

## 7. `SFTConfig` Reference

Defined in `trainer/sfttrainer/sftconfig.py`. Most fields are populated
straight from CLI args / per-dataset overrides in `train.py`; a few are
internal-only (not exposed via CLI):

| Field | Default | Exposed via CLI? | Notes |
|---|---|---|---|
| `total_samples` | — | derived | Per-dataset, after `limit` is applied. |
| `current_example` | — | derived | `row_offset` from `training.yaml`, i.e. resume point. |
| `global_example` | — | derived | Copied from `training.yaml.global_current_example`. |
| `test_train_ratio` | `0.01` | derived | `validation_dataset_limit / total_samples` if the CLI flag is set, else `0.01`. |
| `min_lr_ratio` | `0.8` | no | Floor for `get_lr`'s decay — LR never drops below `0.8 × peak_lr`. |
| `max_test_rows` | `10000` | no | **Unused** in the code shown — `SFTTrainer` materializes the *entire* streaming test split via `list(test_)` regardless of this field. |
| `label_idx` | `-100` | no | Ignore-index for both loss and the assistant-token mask. |
| `batch_size` | `2` | yes | |
| `grad_accum_steps` | `16` | yes | |
| `warmup_steps_ratio` | `0.10` | yes | |
| `checkpoint_dir` | `"checkpoints"` | yes | |
| `resume` | `False` | yes | |
| `resum_same_dataset` | `False` | no | Referenced only in comments/commented-out code in `_train_loop`; has no effect currently. |
| `learning_rate` | `2e-5` | yes (per-dataset) | |
| `logging_steps` | `10` | yes (`--log_interval`) | |
| `eval_steps` | `200` | yes (`--eval_interval`) | |
| `max_length` | `512` | yes (`--max_seq_len`, per-dataset) | |
| `ptdtype` | `torch.float16` | **no** | Hardcoded class attribute, not a dataclass field with CLI wiring — `--disable_amp` does not currently change it. |
| `vram_limit_mb` | `4000` | yes | |
| `max_temp` | `75` | yes | |
| `cooldown_temp` | `60` | yes | |
| `gradient_checkpointing` | `True` | yes | |
| `weight_decay` | `0.1` | yes | |
| `distributed` | `"none"` | yes | |
| `ddp_backend` | `"nccl"` | yes | |
| `ddp_find_unused_parameters` | `False` | no | Set `True` only for architectures where some params legitimately get no gradient on some forward passes (e.g. MoE routing) — otherwise it just disables a DDP optimization. |
| `ddp_timeout_seconds` | `1800` | no | Raise if rank-0's checkpoint save or eval pass is slow enough to trip the collective-op watchdog on other ranks. |
| `ddp_sync_cooldown` | `True` | no | If `True`, every rank barriers after each thermal-cooldown check, so one hot GPU throttles all ranks. Set `False` to let ranks cool down independently (risks a later collective timing out — see the docstring on `check_and_cooldown_gpu`'s call site). |

---

## 8. Pipelines & Preprocessing

`Train._change_template` dispatches on the pipeline name:

| Pipeline | Preprocessing function | Notes |
|---|---|---|
| `IFT` | `process_ift_dataset` (via `Dataset.from_generator` / `IterableDataset.from_generator`) | Expects the resulting examples to already be in the `messages` schema `get_batch` understands. |
| `RFT` | `preprocess_rft` | |
| `TC` | `process_tc` | |
| `IFTC_1` | `process_iftc` | |
| `IFTC_2` | `process_iftc` | Same function as `IFTC_1` — split only matters if you want separate `trained`/`completed` bookkeeping per stage. |
| `WARMUP` | `process_warmup`, then filtered on an `x["valid"]` column and that column dropped | |
| `PT` | `preprocess_text_generation` | Needs `text_column` in the dataset config. |
| *(any other name)* | `preprocess_text_generation` (same as `PT`) | **Silent fallback** — a typo'd pipeline name in `config.yaml` won't error, it'll just get treated as plain-text pretraining. |

---

## 9. Expected Example Schema 

Each dataset row must be one of:

**Chat / SFT:**
1. Text Genration
```json
    {
        "messages": {"text": ".."}
    }
```

2. Supervised Fine Tunning
```json
    {
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
    }
```

3. Reasoning Fine Tunning
```json
    {
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
    }

3. Tool Calling
```json
    {
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
    }
    
```
4. Tool Calling + Reasoning

```json
    {
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
    }
```

---

## 10. Training Loop Internals

### Step structure
- One **optimizer step** = `grad_accum_steps` micro-steps.
- One **outer while-loop iteration** ("epoch loop" in the code, though it's
  really "until this dataset's stream is exhausted") recomputes LR via
  `get_lr(current_example, total_iterations, ...)`, then loops
  micro-batches until a step completes or the whole dataset runs out.

### DDP collective-call discipline (important if you touch this code)
Every rank must call the exact same sequence of `dist.all_reduce` /
`dist.broadcast` calls per micro-step, regardless of which of three cases it
hits locally (out of data / empty-mask batch / normal batch) — otherwise
ranks silently desync their NCCL call count and one rank hangs ~30 min later
on an unrelated collective. The loop enforces this by:
1. Deciding `local_bad_loss` uniformly first (even for the "out of data"
   case, where it's hardcoded `False`).
2. Calling `_any_rank_bad_loss` exactly once per micro-step regardless of case.
3. Always calling exactly one `backward()` — either the real (scaled) loss,
   or a `0.0 * param.sum()` dummy loss for out-of-data/empty-mask ranks, so
   DDP's gradient-sync all-reduce still fires uniformly across ranks.

If you add a new branch to the micro-step logic, preserve this invariant.

### Thermal & VRAM guards
- `check_and_cooldown_gpu` polls `nvidia-smi` before every outer iteration;
  if temp ≥ `max_temp` it sleeps in 10s increments until it drops to
  `cooldown_temp`. With `ddp_sync_cooldown=True`, every rank barriers right
  after, so one hot card pauses the whole job.
- `check_vram_limit` just empties the CUDA cache when allocated memory
  exceeds `vram_limit_mb` — it does not reduce batch size itself (only the
  OOM-catch path below does that).

### OOM recovery
`_train_loop` is wrapped in `try/except RuntimeError`. On a message matching
`"out of memory"`, `"memory limit"`, or `"allowed memory"` (checked on every
rank, then all-reduced with MAX so all ranks agree), it:
1. Halves `BATCH_SIZE` (down to a floor of 1), doubling `GRAD_ACCUM_STEPS` to
   preserve the effective batch size, **or**
2. If batch size is already 1, force-enables `raw_model.gradient_checkpointing`
   and resets to the original batch size/accum steps, **or**
3. If both are already exhausted, re-raises.
4. Then **recurses** into `self._train_loop()` to resume training with the
   new settings.

Any non-OOM `RuntimeError` on any rank is re-raised on every rank (never
silently swallowed on the healthy ranks).

### Checkpoints written
- **`model.pt`** — every `log_interval` examples, and again atomically
  (write to `.tmp`, then `os.replace`) when the dataset's stream is
  exhausted. Contains `model_state_dict`, `optimizer_state_dict`, `step`,
  `best_val_loss` so far, and `run_meta`.
- **`best.pt`** — whenever an eval pass produces a new best `val_loss`.
  Additionally stores `current_example` (global, reduced across ranks).
- On `--resume`, only `model.pt` is loaded (via `pre_resumption_pipeline`,
  i.e. `training.yaml.current_pipeline`) — `best.pt` is informational only
  and never auto-loaded.
- The step-count restore logic is currently **commented out**: only
  `current_example` (floor-divided by `world_size`) is restored on resume;
  `step` always restarts at `0`. This affects the LR schedule's warmup
  bookkeeping if you rely on `step` rather than `current_example` elsewhere.

---

## 11. Logging

Two CSVs per pipeline, header written once by `Trainer.init_csv`:

**`logs_train.csv`** — `global_step, num_examples, loss, perplexity, entropy, mean_token_accuracy, learning_rate, GNorm`, written every `log_interval` examples (or at dataset completion).

**`logs_validation.csv`** — `global_step, num_examples, loss, perplexity, entropy, mean_token_accuracy`, written every `eval_steps` examples (or at dataset completion). Eval currently only ever runs on rank 0's fully-materialized test split.

Console also gets a live `tqdm` bar (main process only) plus periodic
`tqdm.write` summaries with loss/perplexity/grad-norm/step-time/VRAM/temp.

---

## 12. Known Rough Edges (read before you debug)

These are things in the current code that are easy to trip over — flagging
them here so they're not mistaken for intended behavior:

- **`--disable_amp` is parsed but not connected.** `SFTConfig.ptdtype` is a
  hardcoded class attribute (`torch.float16`), not populated from CLI.
- **`--stream_dataset` uses `type=bool`**, which is an argparse footgun:
  `--stream_dataset False` still evaluates to `True` (any non-empty string is
  truthy). Only omit the flag or pass `True`.
- **Unmatched `--model` or pipeline names fall through silently** to a
  default branch rather than erroring — a typo won't fail fast.
- **`Trainer.evaluate_loss` and `Trainer.get_learning_rate` are dead code**
  relative to the current `SFTTrainer` — `evaluate_loss` calls
  `self.get_batch(step, data, data_len, batch_size, max_seq_len, device,
  pipeline, rand=False)`, a signature that doesn't match
  `SFTTrainer.get_batch(DATASET, streaming, count_examples)`. Don't call
  `evaluate_loss` as-is. `get_learning_rate` is likewise unused — the real LR
  schedule is `trainer.util.get_lr`, and the *base* LR comes from
  `SFTConfig.learning_rate` (per-dataset in `config.yaml`), not this method.
- **`SFTConfig.max_test_rows` is unused** — the entire streaming test split
  is materialized on rank 0 regardless of this field's value. For very large
  eval splits, prefer setting `--validation_dataset_limit` instead.
- **`_setup_ddp` is duplicated verbatim** in both `Trainer` and `SFTTrainer`.
  If you change DDP bootstrap behavior, update both.
- **`resum_same_dataset` has no effect** — referenced only in comments /
  commented-out code paths.
- **OOM recovery recurses** rather than looping — a pathological case that
  keeps OOMing after both batch-size halving and gradient checkpointing are
  exhausted will raise (not loop forever), but many consecutive *recoverable*
  OOMs will grow the Python call stack. Not a practical issue at typical
  depths, but worth knowing if you see a `RecursionError` instead of the
  expected final `RuntimeError`.
- **`enforce_gpu_limit` is defined but never called** anywhere in the shown
  code — only the softer `check_vram_limit` (cache-empty on threshold) runs
  automatically.
- **Entry point.** `train.py` does `from trainer import SFTConfig,
  SFTTrainer, ...`, i.e. it's a top-level script, not a module *inside* the
  `trainer` package. Run it as `python train.py ...` (or `python -m train`
  from the repo root); `python -m trainer.train` will only work if your
  actual package layout nests it there.

---

## 13. Adding a New Pipeline

1. Add the pipeline name to `pipeline:` (and its datasets under `dataset:`)
   in `config.yaml`.
2. Add a matching `case "YOUR_PIPELINE":` in `Train._change_template`
   (`train.py`) pointing at a preprocessing function that returns a
   `dataset.map(...)`/generator-based `Dataset`/`IterableDataset` whose rows
   match the `messages` or `text` schema from [§9](#9-expected-example-schema-sfttrainerget_batch).
3. If it needs a new model variant, extend the `match args.model:` block in
   `Train.train` and give it a real `case`, not just relying on the
   catch-all default.

---

## 14. Troubleshooting

| Symptom | Likely cause |
|---|---|
| Job hangs ~30 min into training, only under multi-GPU | A rank took a different code path through a micro-step than the others (see [DDP collective discipline](#ddp-collective-call-discipline-important-if-you-touch-this-code)) — check for a custom exception path that returns/continues without matching every rank's collective calls. |
| Resume silently restarts a dataset from scratch | `training.yaml`'s `pipeline.<name>[i].trained`/`completed` don't match what you expect — check whether `config.yaml`'s dataset `limit` changed since the last run (see [§6](#6-trainingyaml-auto-managed--dont-hand-edit-under---resume)). |
| `--stream_dataset False` still streams | Known argparse `type=bool` issue — see [§12](#12-known-rough-edges-read-before-you-debug). |
| Loss looks unmasked / too high on a chat dataset | Tokenizer has no `apply_chat_template` — falls back to training on the whole flattened conversation, not just assistant turns. |
| `evaluate_loss` raises a `TypeError` | It's calling `get_batch` with an old, incompatible signature — don't use it with the current `SFTTrainer`; eval is otherwise fully handled inline in `_train_loop`. |