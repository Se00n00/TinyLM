from datasets import load_from_disk
from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.normalizers import NFD, StripAccents
from tokenizers.pre_tokenizers import Digits, Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing

class BPETokenizer:
    def __init__(self, path: str, files: None|list[str] = None, vocab_size: int=32768):

        try:
            self.tokenizer = Tokenizer.from_file(path)
        except Exception:
            if not files:
                raise FileNotFoundError(
                    f"Tokenizer file '{path}' not found and no training files provided."
                )
            self._train(files, vocab_size)
            self.tokenizer.save(path)
            
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        self.unk_id = self.tokenizer.token_to_id("[UNK]")
        self.bos_id = self.tokenizer.token_to_id("<|START|>")
        self.eos_id = self.tokenizer.token_to_id("<|END|>")
        
        print("\n--------------------------------------------------------------------------")
        print(f"VOCAB SIZE: {self.vocab_size} | SPECIAL TOKENS: [PAD] [UNK] <|START|> <|END|>")
        print("--------------------------------------------------------------------------\n")

    def _train(self, files: None | list[str], vocab_size: int):

        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

        # Normallization
        self.tokenizer.normalizer = normalizers.Sequence([NFD(), StripAccents()])

        # Pre-tokenization
        self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace()])
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "<|START|>", "<|END|>"])
        
        def batch_iterator():
            for path in files:
                ds = load_from_disk(path)
                for batch in ds.iter(batch_size=1000):
                    yield from batch["text"]

        self.tokenizer.train_from_iterator(batch_iterator(), trainer)
        
        # Post processing
        bos_id = self.tokenizer.token_to_id("<|START|>")
        eos_id = self.tokenizer.token_to_id("<|END|>")
        
        self.tokenizer.post_processor = TemplateProcessing(
            single="<|START|> $A <|END|>",
            special_tokens=[
                ("<|START|>", bos_id),
                ("<|END|>", eos_id),
            ],
        )

    def encode(self, input: list[str] | str):
        if isinstance(input, str):
            return self.tokenizer.encode(input)
        else:
            return self.tokenizer.encode_batch(input)

    def decode(self, input: list[int]):
        return self.tokenizer.decode(input)
