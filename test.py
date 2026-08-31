import math

import torch
from datasets import load_dataset
from tqdm import tqdm

from Model.layers import Config
from Model.models import Model


# ============================================================
# CONFIG
# ============================================================

CHECKPOINT_PATH = "checkpoints/DONE/model.pt"

MAX_LENGTH = 512
STRIDE = 512

device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# TOKENIZER
# ============================================================

# Use the tokenizer you used during training.
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "Se00n00/TinyLM-2"
)

print("Tokenizer vocab:", len(tokenizer))


# ============================================================
# MODEL
# ============================================================

model = Model(
    Config(
        vocab_size=len(tokenizer),
    )
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

print("Model loaded.")
print("Parameters:", sum(p.numel() for p in model.parameters()))


# ============================================================
# WIKITEXT-2
# ============================================================

dataset = load_dataset(
    "Salesforce/wikitext",
    "wikitext-2-raw-v1",
    split="test",
)

text = "\n\n".join(
    x for x in dataset["text"]
    if x.strip()
)

tokens = tokenizer(
    text,
    return_tensors="pt",
).input_ids.to(device)

N = tokens.shape[1]

print("WikiText tokens:", N)


# ============================================================
# EVALUATION
# ============================================================

total_nll = 0.0
total_tokens = 0

for start in tqdm(
    range(0, N, STRIDE),
    desc="WikiText-2",
):

    end = min(
        start + MAX_LENGTH,
        N,
    )

    chunk = tokens[:, start:end]

    if chunk.shape[1] < 2:
        break

    # --------------------------------------------------------
    # Model forward
    # --------------------------------------------------------

    with torch.no_grad():
        logits = model(chunk)

    # --------------------------------------------------------
    # Next-token prediction
    # --------------------------------------------------------

    shift_logits = logits[:, :-1, :]
    targets = chunk[:, 1:]

    log_probs = torch.log_softmax(
        shift_logits,
        dim=-1,
    )

    token_log_probs = log_probs.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    nll = -token_log_probs.sum().item()

    total_nll += nll
    total_tokens += targets.numel()

    if end == N:
        break


# ============================================================
# RESULTS
# ============================================================

mean_nll = total_nll / total_tokens

ppl = math.exp(mean_nll)

print()
print("=" * 60)
print("WikiText-2 Evaluation")
print("=" * 60)
print(f"Tokens evaluated : {total_tokens:,}")
print(f"Mean NLL         : {mean_nll:.6f}")
print(f"Perplexity       : {ppl:.4f}")
print("=" * 60)