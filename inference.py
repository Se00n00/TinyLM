import os
import sys
import argparse
import torch
import torch.nn.functional as F
import time

# from tokenizer import BPETokenizer
# from models import GPT2LMHeadModel, AdvancedLMHeadModel

SYSTEM_PROMPT = """
You are a helpful, conversational AI companion and always provide user with answer
"""

@torch.no_grad()
async def generate(model, tokenizer, user_prompt, max_new_tokens, max_thinking = 150, system_prompt:str|None=None, temperature=1.0, top_k=50, top_p=0.9, max_seq_len:int|None=40000, device="cpu"):
    """
    Generates text from a prompt using KV caching and sampling.
    """
    
    if system_prompt:
        prompt = "<|SYSTEM|>"+system_prompt + "<|USER|>"+ user_prompt + "<|ASSISTANT|>"
    else:
        prompt = user_prompt
    
    # 1. ENCODE PROMPT
    input_ids = tokenizer.encode("<|START|>"+prompt)
    eos_id = tokenizer.encode(tokenizer.eos_token)[0]
    bos_id = tokenizer.encode(tokenizer.bos_token)[0]
    thinking_start = tokenizer.encode("<|THINK|>")[0]
    thinking_end = tokenizer.encode("<|/THINK|>")[0]
    
    if input_ids[-1] == eos_id:
        input_ids.pop()

    if len(input_ids) == 0:
        input_ids = [bos_id]
        
    

    # print(tokenizer.decode(input_ids), end="\n\n", flush=True)

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device) # [1, T]

    generated = list(input_ids)
    curr_input = input_tensor

    num_new_tokens = 0
    
    # 2. GENERATE NEW TOKEN LOOP
    model.eval()
    for i in range(max_new_tokens):
        start_time = time.perf_counter()
        num_new_tokens += 1

        logits = model(curr_input, None)

        # Take only last position
        logits = logits[:, -1, :]

        if temperature == 0.0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        else:
            logits = logits / temperature

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    logits,
                    descending=True,
                    dim=-1
                )

                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1),
                    dim=-1
                )

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = \
                    sorted_indices_to_remove[..., :-1].clone()

                sorted_indices_to_remove[..., 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1,
                    sorted_indices,
                    sorted_indices_to_remove
                )

                logits[indices_to_remove] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

        next_token_id = next_token.item()

        generated.append(next_token_id)

        if next_token_id == eos_id:
            break

        curr_input = torch.tensor(
            [generated],
            dtype=torch.long,
            device=device
        )
        if max_seq_len and len(generated) > max_seq_len:
            break
            
        end_time = time.perf_counter()
        yield start_time-end_time, len(generated), tokenizer.decode([next_token_id])

    
    # print("\n\n TOKENS /SEC : ", num_new_tokens / (end_time - start_time))
    # return tokenizer.decode(generated)

from transformers import AutoTokenizer
from Model.layers import Config
from Model.models import Model
import asyncio

def main():
    parser = argparse.ArgumentParser(description="Generate text using trained 10M GPT-2 or Advanced model.")
    parser.add_argument("--model", type=str, choices=["GPT","Alibi"], default="GPT",
                        help="Model architecture: GPT")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.pt). If not specified, looks in checkpoints/ directory.")
    parser.add_argument("--prompt", type=str, default="The wikitext dataset contains articles about",
                        help="Prompt to generate text from")
    parser.add_argument("--training_name", type=str)
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=[
            "PT",
            "IFT",
            "PFT",
        ],  # Pre-training, Instruction Finetunning, Preference Fine-Tunning
        default="PT",
        help="Pipeline process: pt, it,..",
    )
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature (0.0 for greedy)")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k filtering (0 to disable)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p/nucleus sampling (1.0 to disable)")
    
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("Se00n00/TinyLM-2")
    vocab_size = len(tokenizer)
    print(f"Loaded BPE tokenizer. Vocab size = {vocab_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set checkpoint path if not provided
    checkpoint_file = args.checkpoint
    if checkpoint_file is None:
        checkpoint_file = f"checkpoints/{args.training_name}/{args.model}_{args.pipeline}.pt"

    # Initialize model
    match args.model:
        case 'GPT':
            model = Model(Config(vocab_size=len(tokenizer)))
        
        case 'Alibi':
            model = Model(Config(vocab_size=len(tokenizer)))
        
        case _:
            model = Model(Config(vocab_size=len(tokenizer)))

    # Load weights
    if os.path.exists(checkpoint_file):
        print(f"Loading checkpoint weights from {checkpoint_file}...")
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded successfully")
    else:
        print(f"WARNING: Checkpoint '{checkpoint_file}' not found. Generating with an UNTRAINED model (random weights)!")

    model.to(device)

    # Generate text
    # print(f"\nGenerating {args.max_new_tokens} tokens with prompt: \"{args.prompt}\"")
    print("-" * 60)
    async def generated():
        async for time, num, token in generate(
            model=model,
            tokenizer=tokenizer,
            user_prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            system_prompt=SYSTEM_PROMPT,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_seq_len= 4096,
            device=device
        ):
            print(token,  end="", flush=True)
    
    asyncio.run(generated())
    
    print("\n","-" * 60)

if __name__ == "__main__":
    main()
