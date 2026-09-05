import os
import sys
import json
import argparse
import torch
import safetensors.torch
from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM, AutoTokenizer, AutoConfig


CONFIG_PY_CODE = """import torch
from transformers import PretrainedConfig

class TinyLM2Config(PretrainedConfig):
    model_type = "TinyLM2"

    def __init__(
        self,
        vocab_size=32771,
        d_model=512,
        num_layer=10,
        num_heads=8,
        dropout_prob=0.1,
        ff_hidden_d=819,
        ff_gated=True,
        norm_epsilon=1e-8,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
        use_cache=False,
        **kwargs
    ):
        self.vocab_size = vocab_size
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
from dataclasses import dataclass
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from .configuration_gpt import TinyLM2Config

@dataclass
class Config:
    num_layer: int = 12
    vocab_size: int = 30000

    d_model: int = 512
    num_heads: int = 8
    dropout_prob: float = 0.1

    # Feedforward
    ff_hidden_d: int | None = 819
    ff_gated: bool = True

    # RMS Norm
    norm_epsilon: float = 1e-8
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, epsilon=1e-8):
        super().__init__()

        self.epsilon = epsilon
        self.weights = nn.Parameter(torch.ones(d_model))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        normallized = X * torch.rsqrt(X.pow(2).mean(-1, keepdim=True) + self.epsilon)

        return normallized.type_as(X) * self.weights


class FeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_d: int | None = None,
        gated: bool = True,
        dropout_prob=0.5,
    ):
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
            temp = F.silu(self.gate_proj(X)) * self.up_proj(X)
        else:
            temp = F.silu(self.up_proj(X))

        return self.dropout(self.down_proj(temp))

class Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_prob: float):
        super().__init__()
        
        assert d_model % num_heads == 0, (
            "Number of Heads must be divisible to model dim"
        )
        head_dim = d_model // num_heads
        self.num_heads, self.head_dim = num_heads, head_dim

        self.q_proj = nn.Linear(d_model, head_dim * num_heads, bias=False)
        self.k_proj = nn.Linear(d_model, head_dim * num_heads, bias=False)
        self.v_proj = nn.Linear(d_model, head_dim * num_heads, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.output_dropout = nn.Dropout(dropout_prob)
        self.attention_dropout = nn.Dropout(dropout_prob)
        
        self.register_buffer(
            "alibi_slopes",
            self.get_alibi_slopes(num_heads).view(1, num_heads, 1, 1),
            persistent=False,
        )

    def get_alibi_slopes(self, num_heads: int):
    
        def get_slopes_power_of_2(n):
            start = 2 ** (-2 ** -(math.log2(n) - 3))
            ratio = start
            return torch.tensor(
                [start * ratio ** i for i in range(n)],
                dtype=torch.float32,
            )
    
        if math.log2(num_heads).is_integer():
            return get_slopes_power_of_2(num_heads)
    
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    
        return torch.cat([
            get_slopes_power_of_2(closest_power_of_2),
            self.get_alibi_slopes(2 * closest_power_of_2)[0::2][: num_heads - closest_power_of_2],
        ])


    def forward(self, x: torch.Tensor):
        batch, seqlen, d_model = x.shape

        # [B, L, head_dim * num_heads] <-- [B, L, D_model]
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Reshape <-- View | [B, num_heads, L, head_dim] <-- [B, L, num_heads, head_dim] <-- [B, L, num_heads * head_dim]
        Q = Q.view(batch, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seqlen, self.num_heads, self.head_dim).transpose(1, 2)

        # ---------------- ALiBi ----------------

        pos = torch.arange(seqlen, device=x.device)

        # distance[i,j] = max(i-j, 0)
        # | 0 0 0 0 0 0 0 0 |
        # | 1 0 0 0 0 0 0 0 |
        # | 2 1 0 0 0 0 0 0 |
        # | 3 2 1 0 0 0 0 0 |
        # | 4 3 2 1 0 0 0 0 |
        # | . 4 3 2 1 0 0 0 |
        # | . . 4 3 2 1 0 0 |
        # | P . . 4 3 2 1 0 |
        
        distance = (pos[:, None] - pos[None, :]).clamp(min=0)

        # ``alibi_slopes`` is a non-persistent buffer and ``from_pretrained``
        # clobbers such buffers (yielding garbage ALiBi masks). Recompute the
        # deterministic slopes locally so the mask is correct on any device /
        # dtype / loading path.
        alibi_slopes = self.get_alibi_slopes(self.num_heads).view(
            1, self.num_heads, 1, 1
        ).to(dtype=Q.dtype, device=x.device)

        # [1, H, L, L]
        alibi = -alibi_slopes * distance
        
        # CAUSAL MASK
        causal_mask = torch.tril(torch.ones(seqlen, seqlen)).bool()
        
        combined_mask = alibi.clone()
        combined_mask = combined_mask.to(x.device).masked_fill(~causal_mask.to(x.device), float("-inf"))
        combined_mask = combined_mask.to(dtype=Q.dtype)
        # ---------------------------------------

        
        output = F.scaled_dot_product_attention(
           Q,
           K,
           V,
           attn_mask=combined_mask,
           dropout_p=self.attention_dropout.p if self.training else 0.0,
           is_causal=False,
        )
        # scores = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.head_dim)
        # scores = scores + self.attention_mask[:, :, :seqlen, :seqlen]
        # scores = F.softmax(scores, dim=-1)
        # scores = self.attention_dropout(scores)
        # output = torch.matmul(scores, V)

        # [B, L, D_model] <-- [B, L, num_heads, head_dim] <-- [B, num_heads, L, head_dim]
        output = output.transpose(1, 2).contiguous().view(batch, seqlen, -1)
        return self.output_dropout(self.o_proj(output))


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.norm1 = RMSNorm(config.d_model, config.norm_epsilon)
        self.norm2 = RMSNorm(config.d_model, config.norm_epsilon)
        self.attention = Attention(
            config.d_model, config.num_heads, config.dropout_prob
        )
        self.feedforward = FeedForward(
            config.d_model, config.ff_hidden_d, config.ff_gated, config.dropout_prob
        )

    def forward(self, X: torch.Tensor):
        X = X + self.attention(self.norm1(X))
        X = X + self.feedforward(self.norm2(X))

        return X


class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model) # Tied Head + Embedding Layer
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layer)])
        self.head_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.final_norm = RMSNorm(config.d_model)   
        
        # Tie the weights
        self.head_proj.weight = self.embeddings.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, X: torch.Tensor, kv_cahe = None):
        
        X = self.embeddings(X) # [B, L] --> [B, L, D]
        
        for block in self.blocks:
            X = block(X)
        
        X = self.final_norm(X)
        output = self.head_proj(X)
        
        return output # [B, L, VOCAB_SIZE]
    

class TinyLM2InnerModel(PreTrainedModel):
    config_class = TinyLM2Config
    base_model_prefix = "model"
    
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layer)])
        self.final_norm = RMSNorm(config.d_model, config.norm_epsilon)
        self.head_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.head_proj.weight = self.embeddings.weight
        self.post_init()

    def forward(self, input_ids):
        x = self.embeddings(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.head_proj(x)

class TinyLM2(PreTrainedModel, GenerationMixin):
    config_class = TinyLM2Config
    base_model_prefix = "model"
    _tied_weights_keys = ["model.head_proj.weight"]
    
    def __init__(self, config: TinyLM2Config):
        super().__init__(config)
        self.model = TinyLM2InnerModel(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.model.head_proj

    def set_output_embeddings(self, value):
        self.model.head_proj = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        return_dict=None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else getattr(self.config, "return_dict", True)

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

def convert_checkpoint_to_hf(
    checkpoint_path: str,
    output_dir: str
):
    print("\n==========================================================================")
    print(f" Checkpoint: {checkpoint_path}")
    print(f" Output Dir: {output_dir}")
    print("==========================================================================\n")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load PyTorch Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    args = checkpoint.get("args", None)

    # Infer model config parameters directly from tensor state dict & args
    # print(state_dict.keys())
    vocab_size = state_dict["embeddings.weight"].shape[0]
    d_model = state_dict["embeddings.weight"].shape[1]

    num_layers = len(set(k.split(".")[1] for k in state_dict.keys() if k.startswith("blocks.")))
    ff_hidden_d = state_dict["blocks.0.feedforward.up_proj.weight"].shape[0]
    ff_gated = "blocks.0.feedforward.gate_proj.weight" in state_dict

    # # Determine max_len from state dict buffer if available
    # if "blocks.0.attention.attention_mask" in state_dict:
    #     max_len = state_dict["blocks.0.attention.attention_mask"].shape[2]
    # else:
    #     max_len = getattr(args, "max_seq_len", 1024) if args else 1024

    num_heads = getattr(args, "num_heads", 8) if args else 8
    dropout_prob = getattr(args, "dropout_prob", 0.1) if args else 0.1

    print("  [Config Inferred]")
    print(f"   - Vocab Size:   {vocab_size}")
    print(f"   - Embedding Dim:{d_model}")
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
    
    tokenizer = AutoTokenizer.from_pretrained("Se00n00/TinyLM-2")

    # 3. Write Hugging Face config.json
    config_dict = {
        "architectures": ["TinyLM2"],
        "auto_map": {
            "AutoConfig": "configuration_gpt.TinyLM2Config",
            "AutoModelForCausalLM": "modeling_gpt.TinyLM2"
        },
        "model_type": "TinyLM2",
        "vocab_size": vocab_size,
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
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id,
        "use_cache": False,
    }

    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    # 4. Save Weights (both model.safetensors and pytorch_model.bin)
    formatted_state_dict = {
        f"model.{k}": v.contiguous()
        for k, v in state_dict.items()
        if k != "head_proj.weight"
    }
    if "embeddings.weight" in state_dict:
        formatted_state_dict["model.head_proj.weight"] = state_dict["embeddings.weight"].clone().contiguous()

    safetensors.torch.save_file(formatted_state_dict, os.path.join(output_dir, "model.safetensors"))
    torch.save(formatted_state_dict, os.path.join(output_dir, "pytorch_model.bin"))

    # 5. Save Tokenizer
    tokenizer.save_pretrained(output_dir)

    # Ensure the chat template is persisted into tokenizer_config.json so that
    # lm-eval's ``--apply_chat_template`` (used by the IFT / RFT stages) works.
    tok_cfg_path = os.path.join(output_dir, "tokenizer_config.json")
    tok_cfg = json.load(open(tok_cfg_path, encoding="utf-8"))
    chat_template = tokenizer.chat_template or tok_cfg.get("chat_template")
    if chat_template:
        tok_cfg["chat_template"] = chat_template
        with open(tok_cfg_path, "w", encoding="utf-8") as f:
            json.dump(tok_cfg, f, indent=2, ensure_ascii=False)

    print(f"  [Successfully Saved HF Model & Tokenizer to '{output_dir}']")

    # 6. Verification Load
    cfg = AutoConfig.from_pretrained(output_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(output_dir, config=cfg, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
    print("  [Verification Passed] AutoModelForCausalLM & AutoTokenizer successfully reloaded!")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    prompt = "There existed people who "
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 2. Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    print("Logits shape:", outputs.logits.shape)

    assert outputs.logits.shape[0] == 1
    assert outputs.logits.shape[1] == inputs["input_ids"].shape[1]
    assert outputs.logits.shape[2] == len(tokenizer)

    print("✓ Forward pass works")

    # 3. Generation
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
        )

    text = tokenizer.decode(
        generated[0],
        skip_special_tokens=False,
    )

    print("\nGenerated:")
    print(text)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    print("\n✓ Generation works")
        


def convert(ckpt:str, pipeline:str):
    if os.path.isfile(ckpt):
        convert_checkpoint_to_hf(ckpt, f"TinyLM2{pipeline}")

def parse():
    parser = argparse.ArgumentParser(description="Parse checkpoint")
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--model_pipeline", default="PT", type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse()
    convert(
        ckpt = args.checkpoint_path,
        pipeline = args.model_pipeline
    )