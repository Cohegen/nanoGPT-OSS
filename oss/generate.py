from generator import generate_tokens
from model import model 

import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

prompt = "First Citizen: Before we proceed any further, hear me speak."

prompt_tokens = enc.encode(prompt)

generated = generate_tokens(
    model,
    device="cuda",
    prompt_tokens=prompt_tokens,
    max_tokens=5000
)

text = enc.decode(generated[len(prompt_tokens):])
print(text)