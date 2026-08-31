from transformers import PretrainedConfig


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

        # Transformers generation/cache expects these names
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