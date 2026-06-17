import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking.

    Chef analogy: 8 chefs (heads), each looking at the same recipe (sequence)
    but asking different questions:
      - Q (Query): "What am I looking for right now?"
      - K (Key):   "What does every ingredient offer?"
      - V (Value): "What should I actually contribute?"

    Each chef compares their Q against all K's, softmaxes into percentages,
    then gathers from V weighted by those percentages.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Combined Q, K, V projection: one big linear layer instead of three
        # Projects (B, T, C) → (B, T, 3*C), then we split into Q, K, V
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, return_attention=False):
        B, T, C = x.shape  # batch, sequence length, embed_dim

        # --- Step 1: Project into Q, K, V and split across heads ---
        qkv = self.qkv(x)                                # (B, T, 3*C)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                # (3, B, num_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]                # each: (B, num_heads, T, head_dim)

        # --- Step 2: Scaled dot-product attention ---
        # attn[i,j] = "how much does token i (query) care about token j (key)?"
        attn = (q @ k.transpose(-2, -1)) * self.scale    # (B, num_heads, T, T)

        # --- Step 3: Causal mask — no peeking at future tokens ---
        # Sets upper triangle to -inf so softmax makes them 0
        if mask is not None:
            attn = attn.masked_fill(mask[:, :, :T, :T] == 0, float('-inf'))

        # --- Step 4: Softmax turns scores into percentages ---
        attn_weights = F.softmax(attn, dim=-1)           # (B, num_heads, T, T)
        attn_weights = self.dropout(attn_weights)

        # --- Step 5: Weighted sum of values ---
        # Each token gathers info from all past tokens, weighted by relevance
        out = attn_weights @ v                            # (B, num_heads, T, head_dim)

        # --- Step 6: Merge heads back together ---
        out = out.transpose(1, 2).reshape(B, T, C)        # (B, T, C)
        out = self.out_proj(out)

        if return_attention:
            return out, attn_weights
        return out


class FeedForward(nn.Module):
    """Standard MLP: Linear → GELU → Dropout → Linear → Dropout.

    Chef analogy: After all chefs share findings (attention), each ingredient
    gets individually refined through this station. Expands 4x (think: "let's
    explore more possibilities") then squeezes back ("keep only what matters").

    GELU is smoother than ReLU — it lets small negative values through slightly,
    which helps gradients flow better.
    """

    def __init__(self, embed_dim, expansion_factor=4, dropout=0.1):
        super().__init__()
        hidden_dim = embed_dim * expansion_factor
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """One Transformer block: LayerNorm → Attention (+residual) → LayerNorm → FFN (+residual).

    Chef analogy: One "cooking station" on the assembly line.
      1. Normalize flavors (LayerNorm)
      2. Chefs huddle — share context across all tokens (MultiHeadAttention)
      3. Add back original flavors (residual: "don't lose what we had")
      4. Normalize again
      5. Each ingredient refined alone (FeedForward)
      6. Add back what we had after step 3 (residual)

    Stack 6 of these in a row and you get GPT.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, dropout=dropout)

    def forward(self, x, mask=None, return_attention=False):
        # Attention sub-block: normalize → attend → residual
        attn_out = self.attn(self.ln1(x), mask, return_attention)
        if return_attention:
            attn_out, attn_weights = attn_out
        x = x + attn_out

        # FFN sub-block: normalize → process → residual
        x = x + self.ffn(self.ln2(x))

        if return_attention:
            return x, attn_weights
        return x


class GPT(nn.Module):
    """Minimal GPT-style decoder-only Transformer.

    Chef analogy: The entire kitchen.
      - Token embeddings: ingredient pantry (character → vector)
      - Position embeddings: step numbers on the recipe (position 0,1,2...)
      - TransformerBlocks: 6 cooking stations in a row
      - Final LayerNorm: final taste balance
      - lm_head: serving window — predicts the next ingredient

    Key design choices:
      - Weight tying: same matrix for pantry (embed) and serving (lm_head)
      - Pre-norm: LayerNorm BEFORE attention/FFN (modern GPT style)
      - Learned positions: not fixed sin/cos, learned during training
    """

    def __init__(self, vocab_size, embed_dim=256, num_heads=8,
                 num_layers=6, block_size=256, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.block_size = block_size

        # Ingredient pantry: each character gets a 256-dim vector
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        # Step numbers: position 0, 1, 2... each gets a 256-dim vector
        self.pos_embed = nn.Embedding(block_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # The 6 cooking stations
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Final taste balance + serving window
        self.ln_final = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying: same pantry for storing & serving
        self.lm_head.weight = self.token_embed.weight

        # Initialize all weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, return_attention=False):
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence too long: {T} > {self.block_size}"

        # Token + position embeddings
        tok_emb = self.token_embed(idx)                         # (B, T, C)
        pos = torch.arange(0, T, device=idx.device, dtype=torch.long)
        pos_emb = self.pos_embed(pos)                           # (T, C)
        x = self.dropout(tok_emb + pos_emb)

        # Causal mask: lower triangular, shared across all heads & layers
        mask = torch.tril(torch.ones(1, 1, self.block_size, self.block_size,
                                     device=idx.device))

        # Pass through all transformer blocks
        attentions = []
        for block in self.blocks:
            if return_attention:
                x, attn = block(x, mask, return_attention=True)
                attentions.append(attn)
            else:
                x = block(x, mask)

        # Final norm → project to vocabulary
        x = self.ln_final(x)
        logits = self.lm_head(x)                                # (B, T, vocab_size)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        if return_attention:
            return logits, loss, attentions
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive text generation — one character at a time.

        temperature=1.0: normal randomness
        temperature=0.1: nearly deterministic (boring but safe)
        temperature=2.0: wild and creative
        top_k=40: only pick from top 40 most likely next chars
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to block_size so we don't exceed context window
            idx_cond = idx[:, -self.block_size:]

            logits, _ = self(idx_cond)                          # (B, T, vocab)
            logits = logits[:, -1, :] / temperature             # only last timestep

            # Top-k filtering: zero out everything except top k
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

        self.train()
        return idx

    def count_params(self):
        return sum(p.numel() for p in self.parameters()) / 1e6


def create_causal_mask(T, device='cpu'):
    """Utility: create a causal (lower triangular) mask for a given sequence length."""
    return torch.tril(torch.ones(1, 1, T, T, device=device))
