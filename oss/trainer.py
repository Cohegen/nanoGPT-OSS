from model import Transformer, ModelConfig  # your model file
import tiktoken
import train
from train import TextDataset

# tokenizer
enc = tiktoken.get_encoding("cl100k_base")

# loading  text
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# dataset
dataset = TextDataset(text, enc, seq_len=256)

# Get the actual vocab size from the tokenizer
actual_vocab_size = enc.n_vocab

# model
config = ModelConfig(vocab_size=actual_vocab_size) # Update vocab_size
model = Transformer(config)

# train
train(
    model,
    dataset,
    epochs=1,
    batch_size=32,
    max_iters=5000
)