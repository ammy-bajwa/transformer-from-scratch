"""Visualize attention patterns across layers and heads.

Usage:
    python visualize.py                     # plot all layers for a sample text
    python visualize.py --text "ROMEO:"     # custom text
    python visualize.py --layer 5 --head 3  # specific layer/head
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from model import GPT


def load_model(path='gpt_model.pt'):
    """Load a trained model checkpoint."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)

    model = GPT(
        vocab_size=checkpoint['vocab_size'],
        embed_dim=checkpoint['config']['embed_dim'],
        num_heads=checkpoint['config']['num_heads'],
        num_layers=checkpoint['config']['num_layers'],
        block_size=checkpoint['config']['block_size'],
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    return model, checkpoint['stoi'], checkpoint['itos']


def get_attention(model, stoi, text):
    """Run text through model and collect attention weights from all layers."""
    encoded = [stoi[c] for c in text if c in stoi]
    if not encoded:
        raise ValueError(f"No valid characters in: '{text}'")

    input_ids = torch.tensor([encoded], dtype=torch.long)

    with torch.no_grad():
        _, _, attentions = model(input_ids, return_attention=True)

    # attentions is list of 6 tensors, each (1, 8, T, T)
    tokens = [stoi.get(i, '?') for i in encoded]  # use itos for display
    return attentions, tokens


def plot_single_attention(attn, tokens, layer, head, save_path=None):
    """Plot one attention head as a heatmap."""
    # attn shape: (1, 8, T, T) — take first batch, specific head
    weights = attn[0, head].cpu().numpy()

    # Use itos mapping for display tokens
    # tokens here are integer indices, need itos
    # Actually we pass itos separately
    fig, ax = plt.subplots(figsize=(12, 10))

    # Truncate token labels for readability
    display_tokens = [t.replace('\n', '\\n') for t in tokens]

    sns.heatmap(
        weights,
        xticklabels=display_tokens,
        yticklabels=display_tokens,
        cmap='Blues',
        ax=ax,
        cbar_kws={'label': 'Attention Weight'},
        vmin=0,
        vmax=weights.max(),
    )

    ax.set_xlabel('Key (attended TO) — "What info is available?"')
    ax.set_ylabel('Query (attending FROM) — "What am I looking for?"')
    ax.set_title(f'Attention Weights — Layer {layer + 1}, Head {head + 1}')

    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_all_heads(attentions, tokens, layer, save_path=None):
    """Plot all 8 heads for a given layer in a grid."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    display_tokens = [t.replace('\n', '\\n') for t in tokens]

    for head in range(8):
        weights = attentions[layer][0, head].cpu().numpy()

        sns.heatmap(
            weights,
            xticklabels=display_tokens if head >= 4 else [],
            yticklabels=display_tokens if head % 4 == 0 else [],
            cmap='Blues',
            ax=axes[head],
            cbar=False,
            vmin=0,
            vmax=weights.max(),
        )
        axes[head].set_title(f'Head {head + 1}')
        if head >= 4:
            axes[head].set_xlabel('Key')

    fig.suptitle(f'All 8 Heads — Layer {layer + 1}', fontsize=14, y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_all_layers(attentions, tokens, save_dir=None):
    """Plot head 0 from each layer to show how attention evolves with depth."""
    num_layers = len(attentions)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    display_tokens = [t.replace('\n', '\\n') for t in tokens]

    for layer in range(num_layers):
        weights = attentions[layer][0, 0].cpu().numpy()  # head 0

        sns.heatmap(
            weights,
            xticklabels=display_tokens,
            yticklabels=display_tokens,
            cmap='Blues',
            ax=axes[layer],
            cbar=False,
            vmin=0,
            vmax=weights.max(),
        )
        axes[layer].set_title(f'Layer {layer + 1}, Head 1')
        axes[layer].set_xlabel('Key')
        axes[layer].set_ylabel('Query')

    fig.suptitle('Attention Evolution Across Layers (Head 1)', fontsize=14, y=1.01)
    plt.tight_layout()

    if save_dir:
        path = f'{save_dir}/all_layers.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize Transformer attention patterns')
    parser.add_argument('--model', '-m', type=str, default='gpt_model.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--text', '-t', type=str,
                        default='To be or not to be, that is the question.',
                        help='Text to analyze (keep it short for readability)')
    parser.add_argument('--layer', '-l', type=int, default=None,
                        help='Specific layer to visualize (0-indexed)')
    parser.add_argument('--head', type=int, default=None,
                        help='Specific head to visualize (0-indexed)')
    parser.add_argument('--all-heads', action='store_true',
                        help='Show all 8 heads for a layer')
    parser.add_argument('--all-layers', action='store_true',
                        help='Show all layers (head 0)')
    parser.add_argument('--save', '-s', type=str, default=None,
                        help='Save plot to file instead of showing')
    args = parser.parse_args()

    model, stoi, itos = load_model(args.model)

    # Encode text
    encoded = [stoi[c] for c in args.text if c in stoi]
    tokens = [itos[i] for i in encoded]  # display tokens

    print(f"Analyzing: '{args.text}'")
    print(f"Tokens: {len(tokens)} chars")
    print(f"Model: {len(model.blocks)} layers, 8 heads")

    input_ids = torch.tensor([encoded], dtype=torch.long)

    with torch.no_grad():
        _, _, attentions = model(input_ids, return_attention=True)

    if args.all_heads:
        layer = args.layer if args.layer is not None else len(model.blocks) - 1
        plot_all_heads(attentions, tokens, layer, args.save)
    elif args.all_layers:
        plot_all_layers(attentions, tokens, args.save)
    elif args.layer is not None and args.head is not None:
        plot_single_attention(attentions[args.layer], tokens, args.layer, args.head, args.save)
    else:
        # Default: show each layer one by one
        for layer in range(len(model.blocks)):
            plot_single_attention(attentions[layer], tokens, layer, head=0)


if __name__ == '__main__':
    main()
