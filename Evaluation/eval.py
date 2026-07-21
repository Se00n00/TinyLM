import os
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
import sys
import pickle
import dill
import dataclasses

# 1. Patch dataclasses.fields for datasets compatibility under Python 3.14
original_fields = dataclasses.fields
def patched_fields(class_or_instance):
    try:
        return original_fields(class_or_instance)
    except TypeError:
        return []
dataclasses.fields = patched_fields

# 2. Patch datasets.utils._dill.Pickler._batch_setitems for Python 3.14 dill/pickle compatibility
import datasets.utils._dill
def patched_batch_setitems(self, items, obj=None, *args, **kwargs):
    if getattr(self, "_legacy_no_dict_keys_sorting", False):
        return pickle._Pickler._batch_setitems(self, items, obj)
    try:
        items = sorted(items)
    except Exception:
        from datasets.fingerprint import Hasher
        items = sorted(items, key=lambda x: Hasher.hash(x[0]))
    return pickle._Pickler._batch_setitems(self, items, obj)
datasets.utils._dill.Pickler._batch_setitems = patched_batch_setitems

# 3. Patch datasets.features.features globals to prevent typing List/Dict instantiation errors
import datasets.features.features
datasets.features.features.List = datasets.features.features.Sequence
datasets.features.features.Dict = dict

# 4. Patch ConfigurableTask to dynamically inject num_fewshot for gsm8k (5-shot)
from lm_eval.api.task import ConfigurableTask
original_configurable_task_init = ConfigurableTask.__init__
def patched_configurable_task_init(self, *args, **kwargs):
    original_configurable_task_init(self, *args, **kwargs)
    if self.config.task == 'gsm8k':
        self.config.num_fewshot = 5
        print('--> Dynamic override: set num_fewshot = 5 for gsm8k task!')
ConfigurableTask.__init__ = patched_configurable_task_init

# 5. Patch datasets.load_dataset to inject trust_remote_code=True
import datasets
original_load_dataset = datasets.load_dataset
original_load_dataset_builder = getattr(datasets, "load_dataset_builder", None)

def patched_load_dataset(*args, **kwargs):
    kwargs["trust_remote_code"] = True
    return original_load_dataset(*args, **kwargs)
datasets.load_dataset = patched_load_dataset

if original_load_dataset_builder:
    def patched_load_dataset_builder(*args, **kwargs):
        kwargs["trust_remote_code"] = True
        return original_load_dataset_builder(*args, **kwargs)
    datasets.load_dataset_builder = patched_load_dataset_builder

import json
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from lm_eval import evaluator

# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Define dummy TrainingConfig class in the __main__ namespace to prevent unpickling/pickle errors
class TrainingConfig:
    pass

sys.modules['__main__'].TrainingConfig = TrainingConfig

from model import Config, Model
from Datasets.tokenizer import BPETokenizer

class TokenizerWrapper:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_id
        self.pad_token_id = tokenizer.pad_id
        self.bos_token_id = tokenizer.bos_id
        self.vocab_size = tokenizer.vocab_size

    def encode(self, text, *args, **kwargs):
        res = self._tokenizer.encode(text)
        if hasattr(res, "ids"):
            return res.ids
        return res

    def decode(self, ids, *args, **kwargs):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return self._tokenizer.decode(ids)

class CustomGPTLM(LM):
    def __init__(self, checkpoint_path, tokenizer_path, device="cpu", max_length=512, batch_size=8):
        super().__init__()
        self._device = torch.device(device)
        self.max_length = max_length
        self.batch_size = batch_size

        # Load tokenizer
        raw_tokenizer = BPETokenizer(path=tokenizer_path)
        self._tokenizer_wrapper = TokenizerWrapper(raw_tokenizer)

        # Load checkpoint
        print(f"Loading checkpoint weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self._device, weights_only=False)
        
        # Detect vocab size from checkpoint
        sd = checkpoint["model_state_dict"]
        checkpoint_vocab_size = sd["embedding.token_encoding.weight"].shape[0]
        print(f"Detected checkpoint vocab size: {checkpoint_vocab_size} (tokenizer vocab size: {raw_tokenizer.vocab_size})")

        # Initialize model with exact matching checkpoint vocab size
        config = Config(vocab_size=checkpoint_vocab_size, block_size=max_length)
        self.model = Model(config)
        self.model.load_state_dict(sd)
        self.model.to(self._device)
        self.model.eval()
        print("Checkpoint loaded successfully.")

    @property
    def tokenizer(self):
        return self._tokenizer_wrapper

    @property
    def tokenizer_name(self) -> str:
        return "custom_bpe"

    def _encode_and_slice(self, context, continuation):
        ctx_enc = self.tokenizer.encode(context)
        if ctx_enc and ctx_enc[-1] == self.tokenizer.eos_token_id:
            ctx_enc.pop()
        
        full_enc = self.tokenizer.encode(context + continuation)
        if full_enc and full_enc[-1] == self.tokenizer.eos_token_id:
            full_enc.pop()
            
        # Slicing continuation
        if len(ctx_enc) < len(full_enc) and full_enc[:len(ctx_enc)] == ctx_enc:
            continuation_enc = full_enc[len(ctx_enc):]
        else:
            continuation_enc = full_enc[-max(1, len(full_enc) - len(ctx_enc)):]
            
        # Truncate if exceeds max length
        if len(full_enc) > self.max_length:
            continuation_len = len(continuation_enc)
            if continuation_len >= self.max_length:
                continuation_enc = continuation_enc[-self.max_length + 1:]
                continuation_len = len(continuation_enc)
            full_enc = full_enc[-self.max_length:]
            ctx_len = len(full_enc) - continuation_len
        else:
            ctx_len = len(ctx_enc)
            
        return full_enc, ctx_len, continuation_enc

    def loglikelihood(self, requests: list[Instance], disable_tqdm: bool = False) -> list[tuple[float, bool]]:
        results = []
        
        def process_batch(batch_reqs, current_batch_size):
            nonlocal results
            if not batch_reqs:
                return
            
            try:
                batch_full_enc = []
                batch_ctx_len = []
                batch_cont_enc = []
                
                for req in batch_reqs:
                    context, continuation = req.args
                    full_enc, ctx_len, cont_enc = self._encode_and_slice(context, continuation)
                    batch_full_enc.append(full_enc)
                    batch_ctx_len.append(ctx_len)
                    batch_cont_enc.append(cont_enc)
                    
                max_len = max(len(x) for x in batch_full_enc)
                
                padded_inputs = []
                for x in batch_full_enc:
                    padded_x = x + [self.tokenizer.pad_token_id] * (max_len - len(x))
                    padded_inputs.append(padded_x)
                    
                input_tensor = torch.tensor(padded_inputs, dtype=torch.long, device=self.device)
                
                with torch.no_grad():
                    logits, _ = self.model(input_tensor, None) # shape: [batch, max_len, vocab_size]
                    
                batch_results = []
                for i in range(len(batch_reqs)):
                    L_i = len(batch_full_enc[i])
                    ctx_len_i = batch_ctx_len[i]
                    cont_len_i = len(batch_cont_enc[i])
                    
                    if cont_len_i == 0:
                        batch_results.append((0.0, True))
                        continue
                        
                    start_pos = max(1, ctx_len_i) - 1
                    seq_logits = logits[i, start_pos : L_i - 1, :]
                    seq_targets = torch.tensor(batch_full_enc[i][start_pos + 1 : L_i], dtype=torch.long, device=self.device)
                    
                    log_probs = F.log_softmax(seq_logits, dim=-1)
                    target_log_probs = log_probs[torch.arange(cont_len_i), seq_targets]
                    sum_log_prob = target_log_probs.sum().item()
                    
                    greedy_tokens = seq_logits.argmax(dim=-1)
                    is_greedy = (greedy_tokens == seq_targets).all().item()
                    
                    batch_results.append((sum_log_prob, is_greedy))
                
                results.extend(batch_results)
                
            except torch.OutOfMemoryError as e:
                if current_batch_size <= 1:
                    print("CUDA OOM even with batch_size=1! Falling back to CPU for this batch...")
                    old_device = self._device
                    self._device = torch.device("cpu")
                    self.model.to("cpu")
                    torch.cuda.empty_cache()
                    try:
                        process_batch(batch_reqs, 1)
                    finally:
                        self._device = old_device
                        self.model.to(old_device)
                        torch.cuda.empty_cache()
                else:
                    new_batch_size = max(1, current_batch_size // 2)
                    print(f"CUDA Out of Memory. Reducing batch size from {current_batch_size} to {new_batch_size} and retrying...")
                    torch.cuda.empty_cache()
                    mid = len(batch_reqs) // 2
                    process_batch(batch_reqs[:mid], new_batch_size)
                    process_batch(batch_reqs[mid:], new_batch_size)

        idx = 0
        pbar = tqdm(total=len(requests), disable=disable_tqdm, desc="Evaluating loglikelihood")
        while idx < len(requests):
            batch_reqs = requests[idx : idx + self.batch_size]
            before_len = len(results)
            process_batch(batch_reqs, self.batch_size)
            after_len = len(results)
            num_processed = after_len - before_len
            pbar.update(num_processed)
            idx += len(batch_reqs)
        pbar.close()
        return results

    def loglikelihood_rolling(self, requests: list[Instance], disable_tqdm: bool = False) -> list[float]:
        results = []
        for req in tqdm(requests, disable=disable_tqdm, desc="Evaluating loglikelihood_rolling"):
            try:
                text = req.args[0]
                enc = self.tokenizer.encode(text)
                if enc and enc[-1] == self.tokenizer.eos_token_id:
                    enc.pop()
                
                if len(enc) <= 1:
                    results.append(0.0)
                    continue
                    
                chunks = []
                i = 0
                while i < len(enc):
                    chunk = enc[i : i + self.max_length]
                    if len(chunk) <= 1:
                        break
                    chunks.append((i, chunk))
                    i += self.max_length - 1
                
                total_log_prob = 0.0
                for start_pos, chunk in chunks:
                    input_tensor = torch.tensor([chunk], dtype=torch.long, device=self.device)
                    with torch.no_grad():
                        logits, _ = self.model(input_tensor, None)
                    
                    pred_logits = logits[0, :-1, :]
                    targets = torch.tensor(chunk[1:], dtype=torch.long, device=self.device)
                    
                    log_probs = F.log_softmax(pred_logits, dim=-1)
                    target_log_probs = log_probs[torch.arange(len(targets)), targets]
                    total_log_prob += target_log_probs.sum().item()
                    
                results.append(total_log_prob)
            except torch.OutOfMemoryError:
                print("CUDA OOM in loglikelihood_rolling! Falling back to CPU...")
                old_device = self._device
                self._device = torch.device("cpu")
                self.model.to("cpu")
                torch.cuda.empty_cache()
                try:
                    res = self.loglikelihood_rolling([req], disable_tqdm=True)
                    results.extend(res)
                finally:
                    self._device = old_device
                    self.model.to(old_device)
                    torch.cuda.empty_cache()
        return results

    def generate_until(self, requests: list[Instance], disable_tqdm: bool = False) -> list[str]:
        results = []
        for req in tqdm(requests, disable=disable_tqdm, desc="Generating text"):
            try:
                context, gen_kwargs = req.args
                until = gen_kwargs.get("until", [])
                if isinstance(until, str):
                    until = [until]
                max_gen_toks = gen_kwargs.get("max_gen_toks", self.max_length)
                if max_gen_toks is None:
                    max_gen_toks = self.max_length
                    
                ctx_enc = self.tokenizer.encode(context)
                if ctx_enc and ctx_enc[-1] == self.tokenizer.eos_token_id:
                    ctx_enc.pop()
                    
                generated = list(ctx_enc)
                gen_text = ""
                for _ in range(max_gen_toks):
                    if len(generated) >= self.max_length:
                        break
                        
                    input_tensor = torch.tensor([generated], dtype=torch.long, device=self.device)
                    with torch.no_grad():
                        logits, _ = self.model(input_tensor, None)
                    
                    next_token = logits[0, -1, :].argmax(dim=-1).item()
                    if next_token == self.tokenizer.eos_token_id:
                        break
                        
                    generated.append(next_token)
                    tok_str = self.tokenizer.decode([next_token])
                    gen_text += tok_str
                    
                    stop = False
                    for s in until:
                        if gen_text.endswith(s):
                            gen_text = gen_text[:-len(s)]
                            stop = True
                            break
                    if stop:
                        break
                
                results.append(gen_text)
            except torch.OutOfMemoryError:
                print("CUDA OOM in generate_until! Falling back to CPU...")
                old_device = self._device
                self._device = torch.device("cpu")
                self.model.to("cpu")
                torch.cuda.empty_cache()
                try:
                    res = self.generate_until([req], disable_tqdm=True)
                    results.extend(res)
                finally:
                    self._device = old_device
                    self.model.to(old_device)
                    torch.cuda.empty_cache()
        return results

def get_available_checkpoints():
    checkpoints = []
    # Search in checkpoints/
    if os.path.exists("../checkpoints"):
        for f in os.listdir("../checkpoints"):
            if f.endswith(".pt"):
                checkpoints.append(os.path.join("../checkpoints", f))
    # Search in root
    for f in os.listdir(".."):
        if f.endswith(".pt"):
            checkpoints.append(os.path.join("..", f))
    return sorted(list(set(checkpoints)))

def main():
    parser = argparse.ArgumentParser(description="Evaluate custom GPT checkpoints using lm-eval.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to the checkpoint file. 'all' to evaluate all found checkpoints.")
    parser.add_argument("--tasks", type=str, default="hellaswag,arc_easy,arc_challenge,boolq,piqa,winogrande,mmlu",
                        help="Comma-separated list of tasks to evaluate.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for model inference.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of evaluation samples per task.")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device to use (cuda, cpu). Defaults to auto-detect.")
    parser.add_argument("--tokenizer", type=str, default="../Datasets/sft_tokenizer.json",
                        help="Path to the tokenizer JSON file.")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory to save evaluation results JSON files.")
    
    args = parser.parse_args()

    # Determine device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Find checkpoints
    available_checkpoints = get_available_checkpoints()
    
    if args.checkpoint is None:
        print("No checkpoint specified. Available checkpoints:")
        for idx, cp in enumerate(available_checkpoints):
            print(f"  [{idx}] {cp}")
        if not available_checkpoints:
            print("Error: No checkpoints (.pt files) found in checkpoints/ or root directory.")
            sys.exit(1)
        
        # Default to checkpoints/GPT_IFT.pt if it exists, otherwise the first one
        default_cp = "../checkpoints/GPT_IFT.pt"
        if default_cp not in available_checkpoints:
            default_cp = available_checkpoints[0]
        print(f"Defaulting to: {default_cp}")
        selected_checkpoints = [default_cp]
    elif args.checkpoint.lower() == "all":
        selected_checkpoints = available_checkpoints
        print(f"Evaluating all {len(selected_checkpoints)} checkpoints: {selected_checkpoints}")
    else:
        if not os.path.exists(args.checkpoint):
            alt_path = os.path.join("..", args.checkpoint)
            if os.path.exists(alt_path):
                args.checkpoint = alt_path
            else:
                print(f"Error: Checkpoint file '{args.checkpoint}' not found.")
                sys.exit(1)
        selected_checkpoints = [args.checkpoint]

    tasks_list = [t.strip() for t in args.tasks.split(",") if t.strip()]

    # Make output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load existing summary results if available
    summary_file = os.path.join(args.output_dir, "summary_results.json")
    if os.path.exists(summary_file):
        try:
            with open(summary_file, "r") as f:
                summary_results = json.load(f)
        except Exception:
            summary_results = {}
    else:
        summary_results = {}

    for cp_path in selected_checkpoints:
        cp_name = os.path.basename(cp_path).replace(".pt", "")
        print("\n" + "="*80)
        print(f"EVALUATING CHECKPOINT: {cp_path} ({cp_name})")
        print("="*80 + "\n")

        try:
            output_file = os.path.join(args.output_dir, f"results_{cp_name}.json")
            existing_results = {}
            if os.path.exists(output_file):
                try:
                    with open(output_file, "r") as f:
                        existing_results = json.load(f)
                    print(f"Found existing results file: {output_file}")
                except Exception as e:
                    print(f"Could not load existing results file {output_file}: {e}")

            completed_tasks = []
            if existing_results and "results" in existing_results:
                completed_tasks = list(existing_results["results"].keys())
                print(f"Already completed tasks: {completed_tasks}")

            remaining_tasks = [t for t in tasks_list if t not in completed_tasks]
            
            if not remaining_tasks:
                print(f"All requested tasks for {cp_name} are already evaluated! Skipping simple_evaluate.")
                if cp_name not in summary_results:
                    cp_metrics = {}
                    for task_name, task_results in existing_results["results"].items():
                        metrics = {k: v for k, v in task_results.items() if not k.endswith("_stderr")}
                        cp_metrics[task_name] = metrics
                    summary_results[cp_name] = cp_metrics
                continue

            lm_instance = CustomGPTLM(
                checkpoint_path=cp_path,
                tokenizer_path=args.tokenizer,
                device=device,
                max_length=512,
                batch_size=args.batch_size
            )

            print(f"Running simple_evaluate on remaining tasks: {remaining_tasks}")
            results = evaluator.simple_evaluate(
                model=lm_instance,
                tasks=remaining_tasks,
                limit=args.limit,
                batch_size=args.batch_size,
                confirm_run_unsafe_code=True
            )

            # Merge with existing results
            if existing_results:
                if "results" not in existing_results:
                    existing_results["results"] = {}
                existing_results["results"].update(results.get("results", {}))

                if "configs" not in existing_results:
                    existing_results["configs"] = {}
                existing_results["configs"].update(results.get("configs", {}))

                if "versions" not in existing_results:
                    existing_results["versions"] = {}
                existing_results["versions"].update(results.get("versions", {}))
                
                existing_results["date"] = results.get("date", existing_results.get("date"))
                results = existing_results

            # Extract metrics of interest
            cp_metrics = summary_results.get(cp_name, {})
            for task_name, task_results in results["results"].items():
                metrics = {k: v for k, v in task_results.items() if not k.endswith("_stderr")}
                cp_metrics[task_name] = metrics

            summary_results[cp_name] = cp_metrics

            # Save full results JSON
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Saved merged results to {output_file}")

        except Exception as e:
            print(f"Error evaluating checkpoint {cp_path}: {e}")
            import traceback
            traceback.print_exc()

    # Print summary tables
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    for cp_name, cp_metrics in summary_results.items():
        print(f"\nCheckpoint: {cp_name}")
        print("-" * 50)
        for task, metrics in cp_metrics.items():
            metrics_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
            print(f"  {task:15s} | {metrics_str}")
    print("="*80 + "\n")

    # Save summary results JSON
    summary_file = os.path.join(args.output_dir, "summary_results.json")
    with open(summary_file, "w") as f:
        json.dump(summary_results, f, indent=2, default=str)
    print(f"Saved summary results to {summary_file}")

if __name__ == "__main__":
    main()