import argparse

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    GenerationMixin
)
from transformers.modeling_outputs import CausalLMOutput

from Model.layers import Config
from Model.models import Model

tokenizer = AutoTokenizer.from_pretrained("Se00n00/TinyLM-2")


class TinyLM2Config(PretrainedConfig):
    model_type = "tinylm2"

    def __init__(
        self,
        num_layers=12,
        vocab_size=30000,
        d_model=512,
        num_heads=8,
        dropout_prob=0.1,
        ff_hidden_d=819,
        ff_gated=True,
        norm_epsilon=1e-8,
        tie_word_embeddings=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads

        self.num_hidden_layers = num_layers
        self.hidden_size = d_model
        self.num_attention_heads = num_heads

        self.dropout_prob = dropout_prob
        self.ff_hidden_d = ff_hidden_d
        self.ff_gated = ff_gated
        self.norm_epsilon = norm_epsilon
        self.tie_word_embeddings = tie_word_embeddings

        self.is_decoder = True
        self.is_encoder_decoder = False

class TinyLM2(PreTrainedModel, GenerationMixin):
    config_class = TinyLM2Config

    _tied_weights_keys = {
        "model.head_proj.weight": "model.embeddings.weight"
    }

    def __init__(self, config):
        super().__init__(config)

        self.model = Model(
            Config(
                vocab_size=config.vocab_size,
            )
        )

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
        input_ids,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        logits = self.model(input_ids)

        loss = None

        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutput(
            loss=loss,
            logits=logits,
        )

from hf_model.configuration_tinylm2 import TinyLM2Config
from hf_model.modeling_tinylm2 import TinyLM2

def main(checkpoint_path: str, pipeline: str):

    tokenizer = AutoTokenizer.from_pretrained(
        "Se00n00/TinyLM-2"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    config = TinyLM2Config(
        vocab_size=len(tokenizer)
    )

    model = TinyLM2(config)

    model.model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.tie_weights()

    # THIS is important
    TinyLM2Config.register_for_auto_class()
    TinyLM2.register_for_auto_class(
        "AutoModelForCausalLM"
    )

    output_dir = f"./TinyLM2{pipeline}"

    model.save_pretrained(
        output_dir,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(output_dir)
    


def test(checkpoint_path: str, pipeline: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = f"./TinyLM2{pipeline}"

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    model = model.to(device)
    model.eval()

    # 1. Tokenization

    prompt = "tell me about blackholes !"
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


def parse():
    parser = argparse.ArgumentParser(description="Parse checkpoint")
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--model_pipeline", default="PT", type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()
    main(args.checkpoint_path, args.model_pipeline)
    test(args.checkpoint_path, args.model_pipeline)
