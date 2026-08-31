from datasets import load_dataset, load_dataset_builder
from transformers import AutoTokenizer
from tqdm import tqdm
import time
import os

TARGET_TOKENS = 250_000_000

token = os.getenv("HF_TOKEN")
if not token:
    print("\nWARNING ! Please Use Huggingface Token WhenEver Required !\n")

tokenizer = AutoTokenizer.from_pretrained("Se00n00/TinyLM-2")


def count_examples_until_tokens(dataset:str, subset:str|None = None, data_dir: str|None = None, split:str ="train", text_column: str = "content", target_tokens:int|None = None, max_retries: int = 5):
    print(f"\n====================== Processing {data_dir} ======================")

    for attempt in range(1, max_retries + 1):
        try:
            load_kwargs = {
                "path": dataset,
                "split": split,
                "streaming": True,
                "token": token,
            }
            if subset is not None:
                load_kwargs["name"] = subset          # for the-stack-v2 style
            if data_dir is not None:
                load_kwargs["data_dir"] = data_dir    # for the-stack style
            
            ds = load_dataset(**load_kwargs)

            total_tokens = 0
            num_examples = 0

            pbar = tqdm(desc=f"{dataset}", unit="ex")

            for example in ds:
                text = example.get(text_column) or ""
                if isinstance(text, list) and len(text) > 0:
                    # Check if the 'role' value is invalid
                    if text[0].get('role') not in ['user', 'assistant', 'system']:
                        print("\nWrong chat template !")
                        print(f"Invalid role found: {text[0].get('role')}")
                        break
                       
                # if not isinstance(text, str) or not text.strip():
                #     print("text column name in Wrong !")
                #     break
                if isinstance(text, list):
                    text = {"messages":text}
                # Safer tokenization (avoids the long-sequence warning spam)
                tokens = tokenizer.encode(
                    text,
                    add_special_tokens=False,
                    truncation=False,          # we want full length for counting
                ) if isinstance(text, list) else tokenizer.apply_chat_template(text) 
                total_tokens += len(tokens)
                num_examples += 1

                pbar.update(1)
                pbar.set_postfix({
                    "tokens": f"{total_tokens:,}",
                    "ex": num_examples
                })

                if target_tokens and  total_tokens >= target_tokens:
                    break

            pbar.close()

            print(f"Dataset           : {dataset}")
            print(f"Examples needed  : {num_examples:,}")
            print(f"Total tokens     : {total_tokens:,}")
            print(f"Avg tokens/ex    : {total_tokens / max(num_examples, 1):.1f}")

            return num_examples, total_tokens

        except Exception as e:
            print(f"\n[Attempt {attempt}/{max_retries}] Error on {data_dir}: {type(e).__name__}: {e}")
            if attempt < max_retries:
                sleep_time = 5 * attempt
                print(f"Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                print(f"Failed after {max_retries} attempts. Skipping {data_dir}.")
                return None, None

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Calculate Tokens / Get Amount of tokens of huggingface dataset"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        help="Name of Huggingface Dataset: ex: HuggingFaceFW/fineweb",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="does it contains data dir ?",
    )
    parser.add_argument(
        "--subset",
        type=str,
        help="does it contains subset ?",
    )
    parser.add_argument(
        "--split",
        type=str,
        default = "train",
        help="split if any: ex: train",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        help="which column to encode and count !",
    )
    parser.add_argument(
        "--target_tokens",
        type=int,
        help="which column to encode and count !",
    )
    
    args = parser.parse_args()
    
    data = {"dataset": args.dataset, "text_column": args.text_column, "split": args.split, "target_tokens":args.target_tokens}
    if args.subset:
        data['subset'] = args.subset
    if args.data_dir:
        data['data_dir'] = args.data_dir
        
    # {"data_dir": "data/json",          "text_column": "content"},
    n_ex, n_tok = count_examples_until_tokens(**data)
    print(f"\nEXAMPLES: {n_ex} | TOKENS: {n_tok}")



# USAGE: With Target Tokens
# python -m misc.calculate_tokens \
#   --dataset bigcode/the-stack \
#   --data_dir bigcode/the-stack \
#   --text_column content \
#   --target_tokens 100000000
# 
# USAGE: Without Target Tokens
# python -m misc.calculate_tokens \
#   --dataset HuggingFaceTB/smollm-corpus \
#   --subset cosmopedia-v2 \
#   --text_column text