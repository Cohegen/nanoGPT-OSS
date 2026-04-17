import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

def encode(text):
    return enc.encode(text)

def decode(tokens):
    return enc.decode(tokens)


## Dataset loader
import torch

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, text, tokenizer, seq_len):
        self.tokens = tokenizer.encode(text)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        x = self.tokens[idx:idx+self.seq_len]
        y = self.tokens[idx+1:idx+self.seq_len+1]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long)
        )

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import matplotlib.pyplot as plt

def train(
    model,
    dataset,
    epochs=1,
    batch_size=32,
    lr=2e-4,
    device="cuda",
    max_iters=10000,
    warmup_steps=2000
):
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    scaler = torch.cuda.amp.GradScaler()

    #  tracking lists
    losses = []
    accuracies = []

    step = 0

    for epoch in range(epochs):
        for x, y in dataloader:
            if step >= max_iters:
                break

            x = x.to(device)
            y = y.to(device)

            # LR warmup + cosine decay
            if step < warmup_steps:
                lr_scale = step / warmup_steps
            else:
                progress = (step - warmup_steps) / (max_iters - warmup_steps)
                lr_scale = 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr * lr_scale

            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1)
                )

            # 📊 compute accuracy
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == y).float().mean().item()

            losses.append(loss.item())
            accuracies.append(acc)

            scaler.scale(loss).backward()

            # gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

            if step % 100 == 0:
                print(f"step {step} | loss {loss.item():.4f} | acc {acc:.4f}")

            step += 1

    # plotting after training
    plot_metrics(losses, accuracies)

    return losses, accuracies


def plot_metrics(losses, accuracies):
    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.show()

    plt.figure()
    plt.plot(accuracies)
    plt.title("Training Accuracy")
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.show()