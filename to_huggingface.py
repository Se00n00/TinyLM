import argparse
import json
import os

import safetensors.torch
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)

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

MODELING_PY_CODE = """
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from .configuration_gpt import GPTConfig

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

def convert(args):
    checkpoint_path = args.checkpoint_path

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: '{checkpoint_path}'")

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    args = checkpoint.get("args", None)
    tokenizer = AutoTokenizer.from_pretrained("Se00n00/TinyLM-2")
    vocab_size = state_dict["embedding.token_encoding.weight"].shape[0]
    d_model = state_dict["embedding.token_encoding.weight"].shape[1]
    block_size = state_dict["embedding.position_encoding.weight"].shape[0]

    num_layers = len(
        set(k.split(".")[1] for k in state_dict.keys() if k.startswith("blocks."))
    )
    ff_hidden_d = state_dict["blocks.0.feedforward.up_proj.weight"].shape[0]
    ff_gated = "blocks.0.feedforward.gate_proj.weight" in state_dict

    # Determine max_len from state dict buffer if available
    if "blocks.0.attention.attention_mask" in state_dict:
        max_len = state_dict["blocks.0.attention.attention_mask"].shape[2]
    else:
        max_len = getattr(args, "max_seq_len", 1024) if args else 1024

    num_heads = getattr(args, "num_heads", 8) if args else 8
    dropout_prob = getattr(args, "dropout_prob", 0.1) if args else 0.1

    # 2. Write Standalone HF Configuration and Modeling Modules
    with open(
        os.path.join(args.output_dir, "configuration_gpt.py"), "w", encoding="utf-8"
    ) as f:
        f.write(CONFIG_PY_CODE)

    with open(
        os.path.join(args.output_dir, "modeling_gpt.py"), "w", encoding="utf-8"
    ) as f:
        f.write(MODELING_PY_CODE)

    with open(os.path.join(args.output_dir, "__init__.py"), "w", encoding="utf-8") as f:
        f.write("")

    # 3. Write Hugging Face config.json
    config_dict = {
        "architectures": ["GPTForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_gpt.GPTConfig",
            "AutoModelForCausalLM": "modeling_gpt.GPTForCausalLM",
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
        "torch_dtype": "float32",
    }

    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    # 4. Save Weights (both model.safetensors and pytorch_model.bin)
    formatted_state_dict = {f"model.{k}": v.contiguous() for k, v in state_dict.items()}

    safetensors.torch.save_file(
        formatted_state_dict, os.path.join(args.output_dir, "model.safetensors")
    )
    torch.save(formatted_state_dict, os.path.join(args.output_dir, "pytorch_model.bin"))

    # 6. Verification Load
    try:
        cfg = AutoConfig.from_pretrained(args.output_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.output_dir, config=cfg, trust_remote_code=True
        )
        tok = AutoTokenizer.from_pretrained(args.output_dir, trust_remote_code=True)
        
        sample_chat = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        formatted_chat = tok.apply_chat_template(
            sample_chat, tokenize=False, add_generation_prompt=True
        )
        print(
            f"  [Chat Template Verified] Formatted Output:\n   {repr(formatted_chat)}"
        )
        print(
            f"  [Verification Passed] AutoModelForCausalLM & AutoTokenizer successfully reloaded!"
        )
    except Exception as e:
        print(f"  [Verification Warning] {e}")


if __name__ == "__main__":
    args = parse()

    convert(args)