import torch 
import torch.nn as nn
@torch.no_grad()
def generate_tokens(model, device, prompt_tokens, max_tokens=100000, temperature=0.8, top_k=50):
    model.eval()

    tokens = list(prompt_tokens)

    for i in range(max_tokens):

        x = torch.tensor(tokens[-512:], dtype=torch.long, device=device).unsqueeze(0)

        logits = model(x)[:, -1, :]

        # temperature
        logits = logits / temperature

        # top-k filtering
        topk_vals, topk_idx = torch.topk(logits, top_k, dim=-1)
        probs = torch.softmax(topk_vals, dim=-1)

        next_token = topk_idx[0, torch.multinomial(probs, 1).item()].item()

        tokens.append(next_token)

        if i % 1000 == 0:
            print(f"generated {i} tokens...")

    return tokens
