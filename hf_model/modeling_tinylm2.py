import torch

from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutput

from .configuration_tinylm2 import TinyLM2Config

from Model.layers import Config
from Model.models import Model


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

    def _tie_weights(self):
        self.model.head_proj.weight = self.model.embeddings.weight

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