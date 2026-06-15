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
