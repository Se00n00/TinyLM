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
