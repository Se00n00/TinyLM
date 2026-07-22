import os
import sys
import json
import argparse
import torch
import safetensors.torch
from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM, AutoTokenizer, AutoConfig

from Datasets.tokenizer import BPETokenizer

CONFIG_PY_CODE = """import torch
from transformers import PretrainedConfig

class GPTConfig(PretrainedConfig):
    model_type = "little_parrot"

    def __init__(
        self,
        vocab_size=32771,
        block_size=512,
        max_len=1024,
        d_model=512,
        num_layer=10,
        num_heads=8,
        dropout_prob=0.1,
        ff_hidden_d=2048,
        ff_gated=True,
        norm_epsilon=1e-8,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
        use_cache=False,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.max_position_embeddings = block_size
        self.n_positions = block_size
        self.model_max_length = block_size
        self.max_len = max_len
        self.d_model = d_model
        self.hidden_size = d_model
        self.num_layer = num_layer
        self.num_hidden_layers = num_layer
        self.num_heads = num_heads
        self.num_attention_heads = num_heads
        self.dropout_prob = dropout_prob
        self.ff_hidden_d = ff_hidden_d
        self.ff_gated = ff_gated
        self.norm_epsilon = norm_epsilon
        self.use_cache = use_cache
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
"""

MODELING_PY_CODE = """import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from .configuration_gpt import GPTConfig

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
        self.weights = nn.Parameter(torch.ones(d_model))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        normalized = X * torch.rsqrt(X.pow(2).mean(-1, keepdim=True) + self.epsilon)
        return normalized.type_as(X) * self.weights

class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_d: int | None = None, gated: bool = True, dropout_prob=0.1):
        super().__init__()
        if not hidden_d:
            hidden_d = 4 * d_model
        self.is_gated = gated
        self.up_proj = nn.Linear(d_model, hidden_d, bias=False)
        if gated:
            self.gate_proj = nn.Linear(d_model, hidden_d, bias=False)
        self.down_proj = nn.Linear(hidden_d, d_model, bias=False)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.is_gated:
            temp = F.silu(self.up_proj(X)) * self.gate_proj(X)
        else:
            temp = F.silu(self.up_proj(X))
        return self.dropout(self.down_proj(temp))

class Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_len: int, dropout_prob: float):
        super().__init__()
        head_dim = d_model // num_heads
        self.num_heads, self.head_dim = num_heads, head_dim
        self.q_proj = nn.Linear(d_model, head_dim * num_heads, bias=False)
        self.k_proj = nn.Linear(d_model, head_dim * num_heads, bias=False)
        self.v_proj = nn.Linear(d_model, head_dim * num_heads, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention_dropout = nn.Dropout(dropout_prob)
        self.output_dropout = nn.Dropout(dropout_prob)
        self.register_buffer(
            "attention_mask",
            torch.triu(torch.full((1, 1, max_len, max_len), float("-inf")), diagonal=1)
        )

    def forward(self, x: torch.Tensor):
        batch, seqlen, d_model = x.shape
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        Q = Q.view(batch, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = scores + self.attention_mask[:, :, :seqlen, :seqlen]
        scores = F.softmax(scores, dim=-1)
        scores = self.attention_dropout(scores)
        output = torch.matmul(scores, V)
        output = output.transpose(1, 2).contiguous().view(batch, seqlen, -1)
        return self.output_dropout(self.o_proj(output))

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.token_encoding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = nn.Embedding(block_size, d_model)

    def forward(self, X: torch.Tensor):
        # Truncate input sequence if length exceeds max position embeddings block_size
        if X.size(1) > self.block_size:
            X = X[:, :self.block_size]
        seq_len = X.size(1)
        pos = torch.arange(0, seq_len, dtype=torch.long, device=X.device)
        return self.token_encoding(X) + self.position_encoding(pos)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model, config.norm_epsilon)
        self.norm2 = RMSNorm(config.d_model, config.norm_epsilon)
        self.attention = Attention(config.d_model, config.num_heads, config.max_len, config.dropout_prob)
        self.feedforward = FeedForward(config.d_model, config.ff_hidden_d, config.ff_gated, config.dropout_prob)

    def forward(self, X: torch.Tensor):
        X = X + self.attention(self.norm1(X))
        X = X + self.feedforward(self.norm2(X))
        return X

class InnerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = EmbeddingLayer(config.vocab_size, config.block_size, config.d_model)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layer)])
        self.head_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, X: torch.Tensor):
        X = self.embedding(X)
        for block in self.blocks:
            X = block(X)
        output = self.head_proj(X)
        return output

class GPTPreTrainedModel(PreTrainedModel):
    config_class = GPTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

class GPTForCausalLM(GPTPreTrainedModel, GenerationMixin):
    def __init__(self, config: GPTConfig):
        super().__init__(config)
        self.model = InnerModel(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embedding.token_encoding

    def set_input_embeddings(self, value):
        self.model.embedding.token_encoding = value

    def get_output_embeddings(self):
        return self.model.head_proj

    def set_output_embeddings(self, new_embeddings):
        self.model.head_proj = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        return_dict=None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else getattr(self.config, "return_dict", True)
        if input_ids is not None and input_ids.size(1) > self.config.block_size:
            input_ids = input_ids[:, :self.config.block_size]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.config.block_size]
            if labels is not None:
                labels = labels[:, :self.config.block_size]

        logits = self.model(input_ids)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        return {"input_ids": input_ids, "attention_mask": attention_mask}
"""


SFT_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "<|SYSTEM|>{{ message['content'] }}"
    "{% elif message['role'] == 'user' %}"
    "<|USER|>{{ message['content'] }}"
    "{% elif message['role'] == 'assistant' %}"
    "<|ASSISTANT|>{{ message['content'] }}<|END|>"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt and (messages|length == 0 or messages[-1]['role'] != 'assistant') %}"
    "<|ASSISTANT|>"
    "{% endif %}"
)


def find_checkpoint(candidates):
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def convert_checkpoint_to_hf(
    checkpoint_path: str,
    tokenizer_path: str,
    output_dir: str,
    model_type_label: str = "Model",
    is_sft: bool = False
):
    print(f"\n==========================================================================")
    print(f" Converting {model_type_label} to Hugging Face Format")
    print(f" Checkpoint: {checkpoint_path}")
    print(f" Tokenizer:  {tokenizer_path}")
    print(f" Output Dir: {output_dir}")
    print(f" Chat Template: {'Enabled' if is_sft else 'None'}")
    print(f"==========================================================================\n")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: '{checkpoint_path}'")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found: '{tokenizer_path}'")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load PyTorch Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    args = checkpoint.get("args", None)

    # Infer model config parameters directly from tensor state dict & args
    vocab_size = state_dict["embedding.token_encoding.weight"].shape[0]
    d_model = state_dict["embedding.token_encoding.weight"].shape[1]
    block_size = state_dict["embedding.position_encoding.weight"].shape[0]

    num_layers = len(set(k.split(".")[1] for k in state_dict.keys() if k.startswith("blocks.")))
    ff_hidden_d = state_dict["blocks.0.feedforward.up_proj.weight"].shape[0]
    ff_gated = "blocks.0.feedforward.gate_proj.weight" in state_dict

    # Determine max_len from state dict buffer if available
    if "blocks.0.attention.attention_mask" in state_dict:
        max_len = state_dict["blocks.0.attention.attention_mask"].shape[2]
    else:
        max_len = getattr(args, "max_seq_len", 1024) if args else 1024

    num_heads = getattr(args, "num_heads", 8) if args else 8
    dropout_prob = getattr(args, "dropout_prob", 0.1) if args else 0.1

    print(f"  [Config Inferred]")
    print(f"   - Vocab Size:   {vocab_size}")
    print(f"   - Embedding Dim:{d_model}")
    print(f"   - Block Size:   {block_size}")
    print(f"   - Max Len:      {max_len}")
    print(f"   - Layers:       {num_layers}")
    print(f"   - Heads:        {num_heads}")
    print(f"   - FF Hidden Dim:{ff_hidden_d} (Gated: {ff_gated})")

    # 2. Write Standalone HF Configuration and Modeling Modules
    with open(os.path.join(output_dir, "configuration_gpt.py"), "w", encoding="utf-8") as f:
        f.write(CONFIG_PY_CODE)

    with open(os.path.join(output_dir, "modeling_gpt.py"), "w", encoding="utf-8") as f:
        f.write(MODELING_PY_CODE)

    with open(os.path.join(output_dir, "__init__.py"), "w", encoding="utf-8") as f:
        f.write("")

    # 3. Write Hugging Face config.json
    config_dict = {
        "architectures": ["GPTForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_gpt.GPTConfig",
            "AutoModelForCausalLM": "modeling_gpt.GPTForCausalLM"
        },
        "model_type": "little_parrot",
        "vocab_size": vocab_size,
        "block_size": block_size,
        "max_position_embeddings": block_size,
        "n_positions": block_size,
        "model_max_length": block_size,
        "max_len": max_len,
        "d_model": d_model,
        "hidden_size": d_model,
        "num_layer": num_layers,
        "num_hidden_layers": num_layers,
        "num_heads": num_heads,
        "num_attention_heads": num_heads,
        "dropout_prob": dropout_prob,
        "ff_hidden_d": ff_hidden_d,
        "ff_gated": ff_gated,
        "norm_epsilon": 1e-8,
        "pad_token_id": 0,
        "bos_token_id": 2,
        "eos_token_id": 3,
        "use_cache": False,
        "torch_dtype": "float32"
    }

    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    # 4. Save Weights (both model.safetensors and pytorch_model.bin)
    formatted_state_dict = {f"model.{k}": v.contiguous() for k, v in state_dict.items()}

    safetensors.torch.save_file(formatted_state_dict, os.path.join(output_dir, "model.safetensors"))
    torch.save(formatted_state_dict, os.path.join(output_dir, "pytorch_model.bin"))

    # 5. Convert and Save Tokenizer
    bpe_tokenizer = BPETokenizer(tokenizer_path)
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=bpe_tokenizer.tokenizer,
        bos_token="<|START|>",
        eos_token="<|END|>",
        pad_token="[PAD]",
        unk_token="[UNK]",
        additional_special_tokens=["<|SYSTEM|>", "<|USER|>", "<|ASSISTANT|>"]
    )

    if is_sft:
        hf_tokenizer.chat_template = SFT_CHAT_TEMPLATE

    hf_tokenizer.model_max_length = block_size
    hf_tokenizer.save_pretrained(output_dir)

    print(f"  [Successfully Saved HF Model & Tokenizer to '{output_dir}']")

    # 6. Verification Load
    try:
        cfg = AutoConfig.from_pretrained(output_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(output_dir, config=cfg, trust_remote_code=True)
        tok = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
        if is_sft:
            sample_chat = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ]
            formatted_chat = tok.apply_chat_template(sample_chat, tokenize=False, add_generation_prompt=True)
            print(f"  [Chat Template Verified] Formatted Output:\n   {repr(formatted_chat)}")
        print(f"  [Verification Passed] AutoModelForCausalLM & AutoTokenizer successfully reloaded!")
    except Exception as e:
        print(f"  [Verification Warning] {e}")


def convert_all(
    pt_ckpt: str | None = None,
    sft_ckpt: str | None = None,
    pt_tokenizer: str | None = None,
    sft_tokenizer: str | None = None,
    pt_out_dir: str = "hf_models/pretrained",
    sft_out_dir: str = "hf_models/sft",
    mode: str = "all"
):
    # Candidate search paths for pre-trained model
    pt_ckpt_path = find_checkpoint([pt_ckpt, "checkpoints/GPT_PT.pt", "GPT_pt.pt", "best_GPT.pt"])
    pt_tok_path = find_checkpoint([pt_tokenizer, "Datasets/tokenizer.json", "model/tokenizer.json"])

    # Candidate search paths for SFT model
    sft_ckpt_path = find_checkpoint([sft_ckpt, "checkpoints/GPT_IFT.pt", "GPT_it.pt", "checkpoints/GPT_PFT.pt"])
    sft_tok_path = find_checkpoint([sft_tokenizer, "Datasets/sft_tokenizer.json", "Datasets/tokenizer.json"])

    if mode in ["all", "pretrained"]:
        if pt_ckpt_path and pt_tok_path:
            convert_checkpoint_to_hf(pt_ckpt_path, pt_tok_path, pt_out_dir, model_type_label="Pre-Trained Model", is_sft=False)
        else:
            print(f"[Warning] Could not find pre-trained checkpoint ({pt_ckpt_path}) or tokenizer ({pt_tok_path})")

    if mode in ["all", "sft"]:
        if sft_ckpt_path and sft_tok_path:
            convert_checkpoint_to_hf(sft_ckpt_path, sft_tok_path, sft_out_dir, model_type_label="SFT / Instruction Model", is_sft=True)
        else:
            print(f"[Warning] Could not find SFT checkpoint ({sft_ckpt_path}) or tokenizer ({sft_tok_path})")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert PyTorch GPT models & tokenizers to Hugging Face format.")
    parser.add_argument("--mode", choices=["all", "pretrained", "sft"], default="all", help="Model conversion mode")
    parser.add_argument("--pretrained_ckpt", type=str, default=None, help="Path to pre-trained model .pt checkpoint")
    parser.add_argument("--sft_ckpt", type=str, default=None, help="Path to SFT model .pt checkpoint")
    parser.add_argument("--pretrained_tokenizer", type=str, default=None, help="Path to pre-training tokenizer.json")
    parser.add_argument("--sft_tokenizer", type=str, default=None, help="Path to SFT tokenizer.json")
    parser.add_argument("--pt_out_dir", type=str, default="hf_models/pretrained", help="Output directory for HF Pre-trained model")
    parser.add_argument("--sft_out_dir", type=str, default="hf_models/sft", help="Output directory for HF SFT model")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_all(
        pt_ckpt=args.pretrained_ckpt,
        sft_ckpt=args.sft_ckpt,
        pt_tokenizer=args.pretrained_tokenizer,
        sft_tokenizer=args.sft_tokenizer,
        pt_out_dir=args.pt_out_dir,
        sft_out_dir=args.sft_out_dir,
        mode=args.mode
    )
