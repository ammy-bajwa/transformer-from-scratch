"""Train the mini-GPT on Shakespeare character-by-character.

Usage:
    python train.py                     # train with default config
    python train.py --steps 5000        # shorter training
    python train.py --lr 1e-3           # different learning rate
"""

import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
from model import GPT


class TextDataset(Dataset):
    """Character-level text dataset.

    Chef analogy: The prep cook who chops Shakespeare into
    bite-sized training chunks. Each chunk is block_size characters.
    """

    def __init__(self, text, block_size):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}  # string → index
        self.itos = {i: ch for i, ch in enumerate(chars)}  # index → string

        data = [self.stoi[c] for c in text]
        self.data = torch.tensor(data, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # x: characters 0..block_size-1
        # y: characters 1..block_size  (shifted by 1 — predict next char)
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y

    def decode(self, ids):
        return ''.join(self.itos[i] for i in ids if i in self.itos)

    def encode(self, text):
        return [self.stoi[c] for c in text]


def get_device():
    """Pick the best available device."""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'  # Apple Silicon
    return 'cpu'


def train(config):
    device = get_device()
    print(f"Using device: {device}")

    # --- Load data ---
    with open('data/input.txt', 'r') as f:
        text = f.read()
    print(f"Loaded {len(text):,} characters")

    dataset = TextDataset(text, config['block_size'])
    print(f"Vocabulary size: {dataset.vocab_size} unique characters")

    train_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=True,
    )

    # --- Create model ---
    model = GPT(
        vocab_size=dataset.vocab_size,
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        block_size=config['block_size'],
        dropout=config['dropout'],
    ).to(device)

    print(f"Model parameters: {model.count_params():.1f}M")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )

    # --- Learning rate scheduler: warmup then cosine decay ---
    def lr_lambda(step):
        warmup = config['warmup_steps']
        if step < warmup:
            return step / warmup  # linear ramp-up
        progress = (step - warmup) / (config['max_steps'] - warmup)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- Training loop ---
    model.train()
    best_loss = float('inf')
    step = 0

    print(f"\n{'='*55}")
    print(f"Training for {config['max_steps']} steps...")
    print(f"{'='*55}")

    while step < config['max_steps']:
        for x, y in train_loader:
            if step >= config['max_steps']:
                break

            x, y = x.to(device), y.to(device)

            # Forward pass
            optimizer.zero_grad()
            logits, loss = model(x, y)

            # Backward pass
            loss.backward()

            # Gradient clipping: prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update weights
            optimizer.step()
            scheduler.step()
            step += 1

            if loss.item() < best_loss:
                best_loss = loss.item()

            # Logging
            if step % 100 == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"Step {step:>6d}/{config['max_steps']} | "
                      f"Loss: {loss.item():.4f} | Best: {best_loss:.4f} | "
                      f"LR: {lr:.2e}")

            # Periodic generation sample
            if step % 2000 == 0:
                model.eval()
                prompt = "QUEEN:"
                context = torch.tensor([dataset.encode(prompt)], device=device)
                generated = model.generate(
                    context,
                    max_new_tokens=80,
                    temperature=0.8,
                    top_k=40,
                )
                sample = dataset.decode(generated[0].tolist())
                print(f"\n{'─'*55}")
                print(f"Sample at step {step}:")
                print(sample)
                print(f"{'─'*55}\n")
                model.train()

    # --- Save checkpoint ---
    torch.save({
        'model_state': model.state_dict(),
        'stoi': dataset.stoi,
        'itos': dataset.itos,
        'vocab_size': dataset.vocab_size,
        'config': config,
    }, 'gpt_model.pt')

    print(f"\n{'='*55}")
    print(f"Model saved to gpt_model.pt")
    print(f"Final loss: {best_loss:.4f}")
    print(f"{'='*55}")

    return model, dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train mini-GPT on Shakespeare')
    parser.add_argument('--steps', type=int, default=10000,
                        help='Number of training steps')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    args = parser.parse_args()

    config = {
        'embed_dim': 256,        # embedding dimension
        'num_heads': 8,          # attention heads
        'num_layers': 6,         # transformer blocks
        'block_size': 256,       # context length
        'dropout': 0.1,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'warmup_steps': 500,
        'max_steps': args.steps,
    }

    model, dataset = train(config)
