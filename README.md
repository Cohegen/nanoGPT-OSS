# nanoGPT-OSS

`nanoGPT-OSS` is a small educational GPT-style language model project. It is useful as a compact reference for the core ideas behind modern decoder-only transformers without the amount of infrastructure found in larger training codebases.

## What This Repository Teaches

This codebase covers the main concepts involved in building and training a causal language model:

### 1. Tokenization and Next-Token Prediction
- Text is converted into integer token IDs with `tiktoken` using the `cl100k_base` vocabulary.
- Training examples are built as shifted token windows:
  - `x`: the current sequence
  - `y`: the same sequence shifted by one token
- The model is trained to predict the next token at every position with cross-entropy loss.

This is the standard autoregressive language modeling setup used by GPT-style models.

### 2. Decoder-Only Transformer Architecture
The model in `oss/model.py` is a decoder-only transformer:
- Token embedding layer
- Stack of transformer blocks
- Final normalization layer
- Language modeling head that maps hidden states back to vocabulary logits

Each block uses residual connections and causal self-attention, which means every token can attend only to earlier tokens in the sequence.

### 3. RMSNorm Instead of LayerNorm
The model uses **RMSNorm** rather than standard LayerNorm.

Conceptually, RMSNorm:
- rescales activations based on their root-mean-square magnitude
- keeps the normalization simpler than LayerNorm
- is common in newer open-weight LLM architectures

This lets the repository demonstrate a normalization strategy often used in LLaMA-style models.

### 4. Rotary Positional Embeddings (RoPE)
Instead of learned positional embeddings, the attention module uses **Rotary Positional Embeddings**.

RoPE is important because it:
- injects position information directly into query and key vectors
- preserves relative position structure better than basic absolute embeddings
- is widely used in modern transformer implementations

This repo also includes a scaling parameter for RoPE, showing the idea of stretching positional behavior for longer contexts.

### 5. Grouped-Query Attention
The attention layer separates:
- `num_attention_heads` for queries
- `num_key_value_heads` for keys and values

That means the implementation demonstrates **Grouped-Query Attention (GQA)**:
- many query heads
- fewer key/value heads
- repeated key/value heads to match the number of query heads

This is a practical optimization used in modern models to reduce memory and compute pressure during attention.

### 6. SwiGLU Feed-Forward Network
The MLP uses a **SwiGLU-style** gating pattern:
- one projection is split into two parts
- one part gates the other through `SiLU`
- the result is projected back to the hidden size

This shows the gated feed-forward design that has largely replaced simpler ReLU-based MLPs in many LLMs.

### 7. Weight Tying
The token embedding matrix and output projection share weights.

This concept, usually called **weight tying**, is important because it:
- reduces parameter count
- is standard in many language models
- links input token representations and output token prediction space

### 8. Stable Training Basics
The training code in `oss/train.py` includes several standard optimization ideas:
- `AdamW` optimizer
- learning-rate warmup
- cosine decay
- gradient clipping
- mixed precision with `torch.cuda.amp`

These are not just implementation details; they are core concepts in training transformer language models reliably.

### 9. Autoregressive Text Generation
`oss/generator.py` demonstrates inference-time sampling:
- feed in a prompt
- run the model to get logits for the next token
- apply temperature scaling
- restrict candidates with top-k sampling
- sample one token and append it
- repeat

This is the core generation loop behind GPT-style text completion.

## Files and Their Roles
- `oss/model.py`: model configuration, RMSNorm, RoPE, attention, MLP, transformer blocks, and the full language model
- `oss/train.py`: tokenizer helpers, dataset creation, training loop, loss computation, optimizer schedule, and metric plotting
- `oss/trainer.py`: simple training entry script
- `oss/generator.py`: token-by-token sampling logic for inference
- `oss/generate.py`: generation entry script
- `dataset/input.txt`: training text corpus

## Summary

If you study this repository carefully, you will encounter the main ideas behind a modern small-scale LLM implementation:
- autoregressive language modeling
- causal self-attention
- rotary position encoding
- grouped-query attention
- RMSNorm
- SwiGLU feed-forward layers
- tied embeddings
- standard transformer training and sampling routines

It is best viewed as a compact learning project for understanding how contemporary GPT-like models are put together end to end.
