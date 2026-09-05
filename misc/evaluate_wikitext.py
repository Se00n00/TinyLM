import argparse
import math

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from Model.layers import Config
from Model.models import Model

# ============================================================
# CONFIG
# ============================================================

MAX_LENGTH = 512
STRIDE = 512

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("Se00n00/TinyLM-2")


def evaluate(ckpt, pipeline):
    model = Model(
        Config(
            vocab_size=len(tokenizer),
        )
    )

    checkpoint = torch.load(
        ckpt,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()

    # ============================================================
    # WIKITEXT-2
    # ============================================================

    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="test",
    )

    text = "\n\n".join(x for x in dataset["text"] if x.strip())

    tokens = tokenizer(
        text,
        return_tensors="pt",
    ).input_ids.to(device)

    N = tokens.shape[1]

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

    return ppl, mean_nll, total_tokens


def parse():
    parser = argparse.ArgumentParser(description="Parse checkpoint")
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--model_pipeline", default="PT", type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()
    ppl, mean_nll, total_tokens = evaluate(
        ckpt=args.checkpoint_path, pipeline=args.model_pipeline
    )
    print(f"""
        ___________________________________________
        |    Model  |    Task   |     Metric      |    Value
        |-----------|-----------|-----------------|------------|
        |   Alibi   |  Wikitext | Word_perplexity |  {ppl:.3f}
        |           |           | Mean NLL        |  {mean_nll:.3f}
        |-----------------------------------------|
    """)
