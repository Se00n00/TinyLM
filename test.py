
from trainer import SFTConfig, SFTTrainer
from transformers import AutoTokenizer
from datasets import load_dataset
from datasets import load_dataset_builder
from Model.layers import Config
from Model.models import Model

if __name__ == '__main__':
    builder = load_dataset_builder("Se00n00/FineWeb-1B")
    total_samples = builder.info.splits["train"].num_examples
    
    tokenizer = AutoTokenizer.from_pretrained("Se00n00/TinyLM-2")
    # print(tokenizer.chat_template)
    model = Model(Config(vocab_size=len(tokenizer)))

    trainer = SFTTrainer(
        training_name="Train",
        model=model,
        tokenizer= tokenizer,
        ds=load_dataset("Se00n00/FineWeb-1B", split="train", streaming=True),
        config = SFTConfig(total_samples=total_samples)
    )
    
    for batch_X, batch_y in trainer.get_batch():
        print(batch_X)
        print(batch_y.shape)
    