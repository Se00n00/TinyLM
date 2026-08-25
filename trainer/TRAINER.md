### Training Fresh New Run
  
  1. Edit `trainer/config.yaml` - Mention Pipeline in Sequence Manner + Dataset and config 
  2. check chat template `_change_template` at `trainer/train.py`  
  
  ```bash
  python -m  trainer.train \
    --training_name PreTraining \
    --batch_size 1 \
    --grad_accum_steps 4 \
    --max_seq_len 30000 \
    --stream_dataset True \
    --warmup_steps_ratio 0.10  # <---------------|
    --validation_dataset_limit 1000  # <--- Use Any of these
    
  ```
### Resuming the Training or Training on New Pipeline
  
  1. Ensure configuration `<Checkpoint Dir>/ <Training_Name>/training.yaml` matches `trainer/config.yaml` 
  2. Ensure `<Checkpoint Dir>/ <Training_Name>/training.yaml`'s pipeline content
  
  ```bash
  python -m  trainer.train \
    --training_name PreTraining \
    --batch_size 1 \
    --grad_accum_steps 4 \
    --max_seq_len 30000 \
    --stream_dataset True
    --warmup_steps_ratio 0.10  # <---------------|
    --validation_dataset_limit 1000 \   # <--- Use Any of these
    --resume
  ```

---
```bash
python -m Trainer.trainer
  --training_name Alibi_pretrain 
```

```bash
python train.py \
  --training_name Alibi_pretrain \
  --model Alibi \
  --batch_size 1 \
  --grad_accum_steps 4 \
  --max_seq_len 30000 \
  --pipeline PFT \
  --resume checkpoints/TinyLM-1-70M_IFT.pt \
  --batch_size 2 \
  --grad_accum_steps 8 \
  --max_seq_len 512
```

LOGGING
  - train loss : [CSV]
  - validation loss : [CSV]
  - learning rate : [CSV]
  - Gradient Norm : [CSV]
  - Iterative Batched Dataset similarity : [CSV]
  - Attention Maps : [MEMMAP]
  - Attention Entropy : [CSV]

---
## Pre-Training Experiments
1. Add new Model in Model/
2. Add Model Initiallization in `Trainer/initiallize.py` and `Trainer/pretrainer.py`
  ```python
  case "New_Model":
      model = New_Model(
          Config(vocab_size=len(tokenizer), new_config=value)
      )
  ```
3. Configure Variables to log or monitor in `Trainer/initiallize.py` and `Trainer/pretrainer.py`

### Training
```bash
python -m Trainer.trainer \
  --training_name Alibi_pretrain \
  --model Alibi \
  --batch_size 4 \
  --grad_accum_steps 4 \
  --max_seq_len 768 \
  --learning_rate 3e-4 \
  --pipeline PT
```

### Inference
```bash
python inference.py \
  --checkpoint checkpoints/Alibi_pretrain/Alibi_PT.pt \
  --prompt "Explain the concept of neural networks in simple terms." \
  --temperature 0.7 \
  --top_k 40 \
  --top_p 0.9
```

VLD LOSS: 3.3547
TEST LOSS: 3.2767431803869607 | TEST PPL: 26.48936099216539

Continuing from step 28601 with best val loss: 3.3547

--------------------------------------------------------------------------
TOKENS/PARAMS:0.6880443483731362 : TRAIN TOKENS 120130675| VALIDATION TOKENS: 252059 | TOKENS/STEPS: 4096| STEPS/EPOCH: 29329
DEVICE: cuda MODEL: Alibi| PARAMETERS:78.826171875| BATCH SIZE:2
----------------------------------------------------------------------------


SFTTrainer 

```python
    for examples in DATASET:
        Y_pred = model(X)
        loss = (Y_pred, Y)
        
        loss.backward()
```