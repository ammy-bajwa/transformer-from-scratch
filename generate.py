"""Generate Shakespeare-style text from a trained model.

Usage:
    python generate.py                          # default: prompt="ROMEO:"
    python generate.py -p "JULIET:" -l 500      # custom prompt, 500 chars
    python generate.py -t 0.5 -k 20             # low temp, tight top-k (safe)
    python generate.py -t 1.5                   # high temp (creative/chaotic)
"""

import torch
import argparse
from model import GPT


def load_model(path='gpt_model.pt'):
    """Load a trained model and its vocabulary mappings."""
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

    print(f"Loaded model: {model.count_params():.1f}M params, "
          f"{len(model.blocks)} layers, "
          f"vocab={checkpoint['vocab_size']}")

    return model, checkpoint['stoi'], checkpoint['itos']


def generate(model, stoi, itos, prompt, max_tokens=200,
             temperature=0.8, top_k=40):
    """Generate text continuation from a prompt."""
    # Encode prompt (skip unknown characters)
    encoded = [stoi[c] for c in prompt if c in stoi]
    if not encoded:
        encoded = [0]  # fallback to first character in vocab

    input_ids = torch.tensor([encoded], dtype=torch.long)
    output_ids = model.generate(input_ids, max_tokens, temperature, top_k)

    return ''.join(itos[i] for i in output_ids[0].tolist() if i in itos)


def interactive_mode(model, stoi, itos):
    """Interactive generation loop."""
    print("\nInteractive mode — type 'quit' to exit.\n")

    while True:
        prompt = input("Prompt> ").strip()
        if prompt.lower() in ('quit', 'exit', 'q'):
            break
        if not prompt:
            prompt = ' '

        output = generate(model, stoi, itos, prompt,
                          max_tokens=200, temperature=0.8, top_k=40)
        print(f"\n{output}\n")
        print("─" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Generate text with a trained mini-GPT')
    parser.add_argument('--prompt', '-p', type=str, default='ROMEO:',
                        help='Starting text prompt')
    parser.add_argument('--length', '-l', type=int, default=300,
                        help='Number of characters to generate')
    parser.add_argument('--temperature', '-t', type=float, default=0.8,
                        help='Temperature (0.1=safe, 1.0=normal, 2.0=wild)')
    parser.add_argument('--topk', '-k', type=int, default=40,
                        help='Top-k sampling (only pick from top k chars)')
    parser.add_argument('--model', '-m', type=str, default='gpt_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Interactive mode — type prompts live')
    args = parser.parse_args()

    model, stoi, itos = load_model(args.model)

    if args.interactive:
        interactive_mode(model, stoi, itos)
        return

    print(f"\nPrompt: {args.prompt}")
    print(f"Temperature: {args.temperature} | Top-k: {args.topk} | "
          f"Length: {args.length}")
    print("─" * 60)

    output = generate(model, stoi, itos, args.prompt,
                      args.length, args.temperature, args.topk)
    print(output)
    print("─" * 60)


if __name__ == '__main__':
    main()
