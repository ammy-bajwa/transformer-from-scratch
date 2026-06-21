# 🧠 Build a Transformer from Scratch — The Complete Beginner's Guide

> **Goal**: Understand and build a working GPT-style language model in PyTorch, character by character, trained on Shakespeare.
>
> **Prerequisites**: Basic Python (functions, classes, loops). That's it.
>
> **Time**: ~4-6 hours if you read + code along. ~2 hours if you just read.
>
> **Outcome**: A 4.8M parameter model that generates Shakespeare-like text, plus a web demo.

---

## Table of Contents

1. [Kitchen Setup](#1-kitchen-setup)
2. [What Are We Building? (The Big Picture)](#2-what-are-we-building-the-big-picture)
3. [Part 1: MultiHeadAttention — The Chef Huddle](#3-part-1-multiheadattention--the-chef-huddle)
4. [Part 2: FeedForward — The Solo Workstation](#4-part-2-feedforward--the-solo-workstation)
5. [Part 3: TransformerBlock — The Assembly Line](#5-part-3-transformerblock--the-assembly-line)
6. [Part 4: GPT — The Full Kitchen](#6-part-4-gpt--the-full-kitchen)
7. [Part 5: Training — Making It Learn](#7-part-5-training--making-it-learn)
8. [Part 6: Generation — Making It Talk](#8-part-6-generation--making-it-talk)
9. [Part 7: Visualization — Looking Inside the Brain](#9-part-7-visualization--looking-inside-the-brain)
10. [Part 8: Web Demo — Making It Interactive](#10-part-8-web-demo--making-it-interactive)
11. [Complete Project Structure](#complete-project-structure)
12. [Common Questions & Debugging](#common-questions--debugging)

---

## 1. Kitchen Setup

### The Chef Analogy (Your Mental Model for This Entire Guide)

```
You're the head chef of a restaurant. Your job:

  INPUT:   A ticker tape of food orders, one ingredient at a time.
           "R-O-M-E-O-:-space-W-h-e-r-e-f-o-r-e..."

  PROCESS: 6 cooking stations. Each station has:
           (a) A chef huddle — everyone shares what they know
           (b) Solo refinement — each ingredient gets processed alone

  OUTPUT:  Predict the next ingredient. "Given 'ROMEO:', 
           what comes next? Probably 'W' for 'Wherefore'."
```

Every technical concept we build maps to a kitchen concept. You'll never get lost.

### What You Need

```bash
# 1. Create project folder
mkdir transformer-from-scratch && cd transformer-from-scratch

# 2. Create a virtual environment (your isolated kitchen)
python3 -m venv venv
source venv/bin/activate

# 3. Install ingredients (libraries)
pip install torch torchvision torchaudio
pip install matplotlib seaborn streamlit

# 4. Download training data (Shakespeare)
mkdir data
curl -o data/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# 5. Initialize git (your time machine for code)
git init
git add . && git commit -m "chore: initialize project"
```

### The File Plan

```
transformer-from-scratch/
├── model.py          ← The brain (4 classes, ~250 lines)
├── train.py          ← Teaching the brain to predict characters
├── generate.py       ← Making the brain talk
├── visualize.py      ← Seeing inside the brain
├── demo.py           ← Interactive web app
└── data/input.txt    ← 1.1 million characters of Shakespeare
```

---

## 2. What Are We Building? (The Big Picture)

### The Problem: Predict the Next Character

```
Input:  "ROMEO:"
Target: "OMEO: "  (shifted by 1 — predict each next character)

The model sees 'R', must predict 'O'
The model sees 'RO', must predict 'M'
The model sees 'ROM', must predict 'E'
...and so on.
```

This is called **autoregressive language modeling** — fancy words for "predict what comes next based on what came before."

### The Architecture at 30,000 Feet

```
   "ROMEO:" (text)
       │
       ▼
   Tokenizer:  'R'→17, 'O'→24, 'M'→22, 'E'→14, 'O'→24, ':'→1
       │
       ▼
   Embedding Layer: each number becomes a 256-dimensional vector
   (Think: each character gets a 256-number "description card")
       │
       ▼
   ┌─────────────────────────────────────────┐
   │  6× TransformerBlock ("cooking station")│
   │  ┌─────────────────────────────────────┐│
   │  │  1. LayerNorm (balance flavors)     ││
   │  │  2. MultiHeadAttention (chef huddle)││
   │  │  3. Add residual (keep the original)││
   │  │  4. LayerNorm (balance again)       ││
   │  │  5. FeedForward (solo refinement)   ││
   │  │  6. Add residual (keep step 3)      ││
   │  └─────────────────────────────────────┘│
   └─────────────────────────────────────────┘
       │
       ▼
   lm_head: 256-dim vector → 65 scores (one per character)
   "I'm 85% sure the next character is 'O'"
```

### The Data Shapes (Your Compass)

Throughout this guide, you'll see shapes in comments. They're your lifeline:

| Letter | Meaning | Example | Analogy |
|--------|---------|---------|---------|
| **B** | Batch size | 32 | Number of recipes cooked simultaneously |
| **T** | Sequence length (Time) | 256 | Number of characters in one recipe |
| **C** | Embedding dimension (Channels) | 256 | How many numbers describe each character |

**Golden rule**: Every layer outputs the same shape `(B, T, C)` it receives. This is why we can stack them.

---

## 3. Part 1: MultiHeadAttention — The Chef Huddle

### The Problem Attention Solves

```
Scenario: You're reading "ROMEO: Wherefore art thou Romeo?"

When you get to the SECOND "Romeo", how do you know it's a name?
→ You remember the FIRST "Romeo" from earlier in the sentence.
→ Your brain "attends to" that earlier occurrence.

That's attention: connecting current position to relevant past positions.
```

### The Formula (Don't Panic)

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Let's break it down term by term:

#### Q (Query): "What am I looking for?"

```
Imagine you're at a library looking for a specific book.
You have a "question" in mind: "I need books about Shakespeare tragedies."

Your Query is that question, encoded as numbers.
For token 7 (the second "Romeo"), its Q says:
  "I'm looking for proper nouns, specifically character names."
```

#### K (Key): "Here's what I offer"

```
Every book in the library has a card in the catalog:
  "I'm about: Shakespeare, tragedy, star-crossed lovers."

Every token in the sequence has a Key:
  Token 0 "R": "I'm the start of a proper noun"
  Token 3 "E": "I'm the end of 'ROMEO'"
  Token 7 "R": Key says "I'm also a proper noun start"
```

#### V (Value): "Here's my actual content"

```
The Key tells you IF the book is relevant.
The Value IS the book's content.

The Value contains the rich information:
  Token 0's V: "I'm character ROMEO, male protagonist, emotional..."
  Token 5's V: "I'm a colon, dialogue follows"
```

#### How They Work Together

```
Step 1: Q @ K^T  (Query compares itself to EVERY Key)

         "Token 7 asks: How relevant is token 0 to what I'm looking for?"
         "Token 7 asks: How relevant is token 1?"
         ...
         "Token 7 asks: How relevant is token 6?"

         Result: a row of numbers called "attention scores"
         [0.92, 0.03, 0.12, 0.85, 0.04, 0.01, 0.45]
            ↑                 ↑
         Very relevant!   Also relevant!
         (it's "ROMEO")    (it's part of "ROMEO")

Step 2: / √d_k  (Scale down so no single score dominates)

         Without scaling: [92, 3, 12, 85, 4, 1, 45]
         With scaling:    [5.1, 0.2, 0.7, 4.7, 0.2, 0.05, 2.5]

Step 3: softmax (Turn into percentages that sum to 1)

         [0.35, 0.05, 0.08, 0.30, 0.05, 0.02, 0.15]
            ↑                    ↑
         35% attention       30% attention
         to token 0          to token 3

Step 4: Weighted sum × V (Gather content weighted by relevance)

         output = 0.35 × V[0] + 0.05 × V[1] + ... + 0.15 × V[6]
         
         "My output is 35% ROMEO-flavored, 30% proper-noun-flavored,
          15% recent-context-flavored."
```

### Why 8 Heads?

```
One chef only sees one pattern. 8 chefs see 8 different patterns:

  Head 0: Specializes in grammar structure
  Head 1: Specializes in character names
  Head 2: Specializes in punctuation rhythm
  Head 3: Tracks verb tenses
  ...etc.

Each head gets 32 dimensions (256 ÷ 8). They work in parallel,
then blend their findings together.
```

### The Causal Mask: No Cheating

```
When predicting token 5, you CAN'T look at tokens 6, 7, 8... 
That would be cheating — you'd just copy the answer.

The mask is a triangle:
        t0  t1  t2  t3  t4  t5  t6
    t0 [ 1   0   0   0   0   0   0 ]  ← token 0 only sees itself
    t1 [ 1   1   0   0   0   0   0 ]  ← can see token 0 and itself
    t2 [ 1   1   1   0   0   0   0 ]
    t3 [ 1   1   1   1   0   0   0 ]
    t4 [ 1   1   1   1   1   0   0 ]
    t5 [ 1   1   1   1   1   1   0 ]

    1 = can see (allowed)
    0 = blocked (replaced with -infinity, softmax makes it 0%)
```

### Walk the Code

Open `model.py` and find the `MultiHeadAttention` class. Let's walk through `forward()`:

```python
def forward(self, x, mask=None, return_attention=False):
    B, T, C = x.shape  
    # B = batch size (how many sequences at once)
    # T = sequence length (how many characters)
    # C = embed_dim (256 numbers per character)

    # STEP 1: One big projection for Q, K, V
    qkv = self.qkv(x)                         # (B, T, 768)
    # 768 = 3 × 256. Split into Q(256) + K(256) + V(256)
    
    # STEP 2: Reshape into 8 heads of 32 dims each
    qkv = qkv.reshape(B, T, 3, 8, 32)        # (B, T, 3, 8, 32)
    qkv = qkv.permute(2, 0, 3, 1, 4)         # (3, B, 8, T, 32)
    q, k, v = qkv[0], qkv[1], qkv[2]         # each (B, 8, T, 32)
    
    # STEP 3: Attention scores
    attn = (q @ k.transpose(-2, -1)) * (32 ** -0.5)  # (B, 8, T, T)
    # q @ k.T: "How much does each token care about every other?"
    # * 1/√32: Scale to keep numbers manageable
    
    # STEP 4: Apply causal mask (hide future)
    if mask is not None:
        attn = attn.masked_fill(mask == 0, float('-inf'))
    
    # STEP 5: Softmax → percentages
    attn_weights = F.softmax(attn, dim=-1)     # (B, 8, T, T)
    
    # STEP 6: Weighted sum of values
    out = attn_weights @ v                     # (B, 8, T, 32)
    
    # STEP 7: Merge heads back
    out = out.transpose(1, 2).reshape(B, T, 256)  # (B, T, 256)
    out = self.out_proj(out)                       # blend heads
    
    return out
```

### Try It Yourself

```python
import torch
from model import MultiHeadAttention

# Create a 256-dim attention with 8 heads
mha = MultiHeadAttention(embed_dim=256, num_heads=8)

# Create fake data: 2 sequences of 10 characters each
x = torch.randn(2, 10, 256)

# Run attention
out = mha(x)
print(f"Input:  {x.shape}")    # (2, 10, 256)
print(f"Output: {out.shape}")  # (2, 10, 256) ← same shape!
```

---

## 4. Part 2: FeedForward — The Solo Workstation

### What It Does

```
After the chef huddle (attention), each token now has context from
the whole sequence. But the information is jumbled.

The FeedForward is where each token goes ALONE to a private workstation
and processes everything it learned:

  256-dim token → expand to 1024 (explore) → GELU filter → squeeze to 256 (keep only what matters)
```

### Why Expand Then Squeeze?

```
Analogy: You're examining a tomato.

Expand (256 → 1024):
  Slice it open. Look at it under a microscope. Examine it from 4 angles.
  "What's the pH? Sugar content? Color spectrum? Texture?"

GELU filter:
  A smart sieve. Keeps positive signals, lets SOME slightly negative 
  ones through. Better than ReLU which kills everything negative.

Squeeze (1024 → 256):
  "What did I actually learn that's useful for predicting the next character?"
  Distill everything back to a compact form.
```

### GELU vs ReLU

```
ReLU:  f(x) = max(0, x)
       "You're either in or out. Negative = dead forever."

GELU:  f(x) = x × Φ(x)  (x times the Gaussian CDF)
       "You're multiplied by the probability you're positive."
       
       x = -0.5 → GELU(-0.5) ≈ -0.15  (small negative, "maybe useful?")
       x = -3.0 → GELU(-3.0) ≈ -0.001 (almost zero, "probably useless")
       x =  2.0 → GELU(2.0)  ≈  1.95  (mostly through, "confidently useful")

This probabilistic filtering helps gradients flow during training.
```

### The Code (model.py → FeedForward)

```python
class FeedForward(nn.Module):
    def __init__(self, embed_dim=256, expansion_factor=4, dropout=0.1):
        super().__init__()
        hidden_dim = embed_dim * expansion_factor  # 256 × 4 = 1024
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),   # 256 → 1024
            nn.GELU(),                          # smart filter
            nn.Dropout(dropout),                # random mute 10%
            nn.Linear(hidden_dim, embed_dim),   # 1024 → 256
            nn.Dropout(dropout),                # random mute 10%
        )
    
    def forward(self, x):
        return self.net(x)
        # Input:  (B, T, 256)
        # Output: (B, T, 256)  ← same shape, richer content
```

### Why Dropout?

```
During training, every time a token goes through FeedForward,
randomly mute 10% of the 1024 intermediate neurons.

  Day 1: Neurons 3, 47, 89 are muted → other neurons compensate
  Day 2: Neurons 12, 55, 91 are muted → different neurons compensate

Result: No single neuron becomes a "crutch." The network builds 
redundancy. Like cross-training — if you only do bench press,
you're weak at everything else.
```

---

## 5. Part 3: TransformerBlock — The Assembly Line

### Bringing Attention and FFN Together

```
One TransformerBlock = one complete "cooking station":

  x → [LayerNorm] → [MultiHeadAttention] → (+x_original) 
    → [LayerNorm] → [FeedForward] → (+x_after_attention)
    → output

Why this order?
  - LayerNorm first (pre-norm): stabilize values before processing
  - Attention: mix information across tokens
  - Residual (+x): "Keep the original recipe notes. If the huddle 
    produced garbage, at least we still have what we started with."
  - LayerNorm again: re-balance after mixing
  - FeedForward: refine each token individually
  - Residual again: "Keep what the huddle taught us."
```

### Why Residuals Are Critical

```
Without residuals:
  Deep networks "forget" the original input → vanishing gradients
  6 layers might work, 12 layers collapse, 96 layers impossible

With residuals:
  Each layer only needs to learn "what to ADD or CORRECT"
  Not "what to become from scratch"
  
  Like editing a document: you don't rewrite from blank page each time.
  You make small edits to the existing draft.

This is why transformers scale to 96+ layers.
```

### LayerNorm: The Taste Balancer

```
After attention mixes flavors, some tokens might be too "loud"
(some dimensions are 100, others are 0.01).

LayerNorm fixes this:
  For each token:
    1. Calculate mean and std of its 256 dimensions
    2. Subtract mean, divide by std → now centered at 0 with spread 1
    3. Scale by learned parameters γ, shift by β
  
  It's like adjusting seasoning: "Too salty? Dilute. Too bland? Add spice."
  But the γ and β are LEARNED — the model figures out the right balance.
```

### The Code (model.py → TransformerBlock)

```python
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)        # pre-attention norm
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)        # pre-FFN norm
        self.ffn = FeedForward(embed_dim, dropout=dropout)

    def forward(self, x, mask=None):
        # Sub-block 1: Attention
        x = x + self.attn(self.ln1(x), mask)     # norm → attend → residual
        
        # Sub-block 2: FeedForward
        x = x + self.ffn(self.ln2(x))            # norm → process → residual
        
        return x
        # Input:  (B, T, 256)
        # Output: (B, T, 256) ← STILL the same shape!
```

### Stacking 6 Blocks

```python
# 6 cooking stations in a row
blocks = [TransformerBlock(256, 8) for _ in range(6)]

# Each block outputs (B, T, 256), next block accepts (B, T, 256)
x = initial_embedding
for block in blocks:
    x = block(x)  # x flows through all 6, getting richer each time
```

---

## 6. Part 4: GPT — The Full Kitchen

### All the Pieces Together

```
The GPT class assembles everything:

  1. Token Embedding:    character → 256-dim vector
  2. Position Embedding: position 0,1,2... → 256-dim vector
  3. Dropout
  4. 6 × TransformerBlock
  5. Final LayerNorm
  6. lm_head: 256-dim → vocab_size scores
  7. Weight initialization
  8. generate() method
```

### Token Embeddings: The Ingredient Pantry

```python
self.token_embed = nn.Embedding(vocab_size=65, embed_dim=256)
# A lookup table: 65 rows, each 256 numbers
# token_embed[17] → the vector for character 'R'
# token_embed[24] → the vector for character 'O'

# Before training: random vectors
# After training: 'R' and 'O' are close because they appear together in "ROMEO"
```

### Position Embeddings: The Recipe Step Numbers

```
"ROMEO" has 'R' at position 0 and 'O' at position 4.
Without position info, the model sees them as identical.

Position embeddings ADD a unique signal for each position:

  pos_embed[0] = [0.01, 0.02, -0.01, ...]   added to 'R' at position 0
  pos_embed[4] = [0.03, -0.01, 0.02, ...]   added to 'O' at position 4

Now the model knows: "An 'R' at position 0 is different from an 'R' at position 50."
```

### Weight Tying: One Pantry for Storage AND Serving

```python
self.token_embed = nn.Embedding(65, 256)   # pantry: character → vector
self.lm_head = nn.Linear(256, 65)           # serving: vector → character scores

# TRICK: Make them share the same weight matrix
self.lm_head.weight = self.token_embed.weight

# Why? The embedding matrix already knows which characters are similar.
# Reusing it for prediction saves parameters and improves training.
```

### The Full Forward Pass

```python
def forward(self, idx, targets=None):
    B, T = idx.shape  # e.g., (32, 256) — 32 sequences of 256 chars
    
    # 1. Embed tokens + positions
    tok_emb = self.token_embed(idx)              # (B, T, 256)
    pos = torch.arange(0, T)                     # [0, 1, 2, ..., 255]
    pos_emb = self.pos_embed(pos)                # (T, 256)
    x = tok_emb + pos_emb                        # (B, T, 256)
    # Note: pos_emb is broadcast across batch dimension
    
    # 2. Create causal mask (shared across all layers)
    mask = torch.tril(torch.ones(1, 1, 256, 256))
    #    [[1,0,0,...],
    #     [1,1,0,...],
    #     [1,1,1,...]]
    
    # 3. Pass through 6 transformer blocks
    for block in self.blocks:
        x = block(x, mask)
    
    # 4. Final norm + project to vocabulary
    x = self.ln_final(x)                         # (B, T, 256)
    logits = self.lm_head(x)                     # (B, T, 65)
    
    # 5. Compute loss (if training)
    if targets is not None:
        loss = F.cross_entropy(
            logits.view(-1, 65),                  # flatten to (B*T, 65)
            targets.view(-1)                      # flatten to (B*T,)
        )
        return logits, loss
    
    return logits, None
```

### Text Generation (The Fun Part)

```
How the model writes Shakespeare one character at a time:

  1. Start with prompt: "ROMEO:" → [17, 24, 22, 14, 24, 1]
  2. Run through model → get 65 scores for "what comes next"
  3. Apply temperature: divide scores by 0.8
     - Low temp (0.1): model picks the "safe" choice (boring)
     - High temp (2.0): model explores wildly (creative gibberish)
  4. Top-k filtering: keep only top 40 scores, zero out rest
  5. Softmax → probability distribution
  6. Sample from this distribution (NOT just pick highest!)
  7. Append chosen character
  8. Repeat steps 2-7 for max_new_tokens

Why sample instead of picking the highest?
  "ROMEO:" → highest probability is ' ' (space) → model generates
  "ROMEO:                      " (spaces forever!)
  
  Sampling introduces controlled randomness → actual text.
```

### The Code (model.py → generate())

```python
@torch.no_grad()  # don't track gradients (faster, less memory)
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    self.eval()
    for _ in range(max_new_tokens):
        # Crop to fit context window
        idx_cond = idx[:, -256:]
        
        # Get predictions
        logits, _ = self(idx_cond)           # (B, T, 65)
        logits = logits[:, -1, :]             # only last position
        logits = logits / temperature         # control randomness
        
        # Top-k filter
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float('-inf')
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)
    
    self.train()
    return idx
```

---

## 7. Part 5: Training — Making It Learn

### The Training Loop (5 Steps Every Time)

```
for each batch of Shakespeare text:
    ┌─────────────────────────────────────────────┐
    │ 1. optimizer.zero_grad()                    │
    │    "Wipe the whiteboard clean"              │
    │                                             │
    │ 2. logits, loss = model(x, y)               │
    │    "Make a prediction and taste-test it"    │
    │                                             │
    │ 3. loss.backward()                          │
    │    "Figure out which spices to adjust"      │
    │                                             │
    │ 4. clip_grad_norm_(model.parameters(), 1.0) │
    │    "Don't let any adjustment be too extreme"│
    │                                             │
    │ 5. optimizer.step()                         │
    │    "Adjust all 4.8 million spice amounts"   │
    └─────────────────────────────────────────────┘
```

### TextDataset: How We Feed the Model

```python
# Shakespeare text (1.1 million characters):
# "First Citizen: Before we proceed any further..."

# We chop it into overlapping windows of 257 characters:
# 
# Window 0: "First Citizen: Before we..."  (chars 0..256)
#   x = chars 0..255   (what model sees)
#   y = chars 1..256   (what model predicts)
#
# Window 1: "irst Citizen: Before we p..."  (chars 1..257)
#   x = chars 1..256
#   y = chars 2..257
#
# This gives us ~1.1 million training examples!

class TextDataset(Dataset):
    def __init__(self, text, block_size=256):
        chars = sorted(list(set(text)))            # find all 65 unique chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}  # char→index
        self.itos = {i: ch for i, ch in enumerate(chars)}  # index→char
        self.data = [self.stoi[c] for c in text]   # entire text as indices
        self.block_size = block_size
    
    def __len__(self):
        return len(self.data) - self.block_size    # ~1.1M examples
    
    def __getitem__(self, idx):
        x = self.data[idx : idx + 256]             # 256 input chars
        y = self.data[idx+1 : idx + 257]           # 256 target chars (shifted)
        return x, y
```

### AdamW Optimizer: The Smart Spice Adjuster

```
Not all parameters learn at the same rate:
  - Embedding weights need big adjustments early
  - Attention weights need careful, small adjustments

AdamW tracks TWO things for each parameter:
  1. Momentum (first moment): "Which direction have we been going?"
     Like a ball rolling downhill — keeps going the same way.
  
  2. Velocity (second moment): "How bumpy is this path?"
     If the gradient keeps flipping (up, down, up, down),
     slow down. If it's consistent, speed up.

  + Weight decay: "Slightly shrink all weights every step."
    Prevents any single weight from dominating.
```

### Learning Rate Schedule: Warmup + Cosine Decay

```
Why warmup?
  At step 0, weights are random. Gradients are huge and chaotic.
  If you start with full learning rate, the model "explodes"
  (loss → infinity, numbers become NaN).

  Solution: Ramp up slowly for 500 steps.
  Step   0: LR = 0.000000
  Step 250: LR = 0.000150
  Step 500: LR = 0.000300  (full speed)

Why cosine decay?
  Early training: big steps to learn the basics (character frequencies)
  Late training: tiny steps to fine-tune (Shakespeare style)
  
  Cosine smoothly transitions from full speed to 10% of full speed.
```

### What Loss Tells You

```
Step    0: Loss = 4.17  (ln(65) = random guessing across 65 chars)
Step  100: Loss = 2.88  ("e" is common, "z" is rare — model learned frequencies)
Step 1000: Loss = 1.80  (learned common words: "the", "and", "thou")
Step 5000: Loss = 1.20  (character names, dialogue structure)
Step10000: Loss = 1.08  (coherent Shakespeare-style text)

Loss = -ln(probability assigned to correct character)
  Loss 4.17 → model gave correct answer 1.5% chance (1/65)
  Loss 1.08 → model gave correct answer 34% chance  (e^1.08 ≈ 2.9, 1/2.9 ≈ 34%)
```

### Running Training

```bash
python train.py

# Custom settings:
python train.py --steps 5000 --lr 1e-3 --batch-size 64
```

Training takes ~2 minutes on Apple Silicon MPS, ~30 minutes on CPU.

---

## 8. Part 6: Generation — Making It Talk

### The Generation Script

```bash
# Basic generation
python generate.py

# Custom prompt
python generate.py -p "JULIET:" -l 500

# Control creativity
python generate.py -t 0.5    # safe, repetitive
python generate.py -t 1.5    # creative, sometimes nonsensical

# Interactive mode
python generate.py -i
```

### Temperature Deep Dive

```
Temperature = 0.1 ("safe mode"):
  Scores become very spread out → softmax picks one very confidently.
  Result: "ROMEO: I am the son of the duke of the duke of the..."
  (Gets stuck in loops — always picks the "safest" next character)

Temperature = 1.0 ("normal"):
  Scores unchanged → natural sampling.
  Result: "ROMEO: I pray thee, gentle Mercutio, let's retire..."
  (Natural variation, realistic Shakespeare)

Temperature = 2.0 ("creative mode"):
  Scores compressed → all options become equally likely.
  Result: "ROMEO: Xqzp!f blart sindle worp hath..."
  (Too random — model ignores what it learned)
```

### Top-K Sampling

```
Without top-k: model considers ALL 65 characters.
  "After 'ROMEO:', next char could be 'X' (0.0001%) or ' ' (85%)."
  The 0.0001% adds up over 500 characters → eventual gibberish.

With top-k=40: only the 40 most likely characters survive.
  The bottom 25 are zeroed out. Much cleaner output.
```

---

## 9. Part 7: Visualization — Looking Inside the Brain

### What Attention Patterns Reveal

```
After training, you can see which characters the model pays attention to:

  Layer 1, Head 0: Mostly diagonal — each character looks at itself.
    "I'm 'e'. I know I'm 'e'."
    
  Layer 3, Head 2: Nearby neighbors.
    "'t' followed by 'h' → this is probably 'the'"
    
  Layer 6, Head 5: Long-range connections.
    "ROMEO appeared 200 characters ago → this response is for him"
```

### Running Visualizations

```bash
# All layers one by one
python visualize.py

# Specific head
python visualize.py --layer 5 --head 3

# All 8 heads in one layer
python visualize.py --all-heads

# Evolution across layers
python visualize.py --all-layers
```

### How to Read the Heatmap

```
        KEYS (attended TO)
        R   O   M   E   O   :
    R [1.0 0.0 0.0 0.0 0.0 0.0]  ← 'R' only sees itself
    O [0.3 0.7 0.0 0.0 0.0 0.0]  ← 'O' sees 'R' (30%) and itself (70%)
Q   M [0.1 0.2 0.7 0.0 0.0 0.0]
U   E [0.1 0.1 0.3 0.5 0.0 0.0]
E   O [0.0 0.1 0.2 0.3 0.4 0.0]  ← 2nd 'O' sees 1st 'O' (10%)
R   : [0.0 0.0 0.1 0.2 0.3 0.4]
S
    Dark blue = 0.0 (no attention)
    Light blue/white = higher attention
    Upper right triangle = always dark (causal mask)
```

---

## 10. Part 8: Web Demo — Making It Interactive

```bash
streamlit run demo.py
```

Opens at `http://localhost:8501` with two tabs:

### Tab 1: Text Generation
- Enter a prompt (e.g., "ROMEO:")
- Adjust temperature, top-k, and max tokens with sliders
- Click "Generate" → see Shakespeare text appear
- Download generated text

### Tab 2: Attention Visualizer
- Enter text to analyze
- Choose which layer (1-6) and head (1-8) to view
- Click "Visualize" → see the attention heatmap
- Dark areas = no attention, light areas = high attention

---

## Complete Project Structure

```
transformer-from-scratch/
│
├── model.py              ← The Brain (250 lines)
│   ├── MultiHeadAttention    Q·K^T/√dk, softmax, causal mask, 8 heads
│   ├── FeedForward           Linear→GELU→Linear, expand 4x
│   ├── TransformerBlock      Norm→Attn→Residual→Norm→FFN→Residual
│   └── GPT                   Embeddings, 6 blocks, lm_head, generate()
│
├── train.py              ← The Teacher (200 lines)
│   ├── TextDataset           Character-level data loader
│   └── train()               AdamW, warmup+cosine LR, gradient clipping
│
├── generate.py           ← The Voice (100 lines)
│   └── CLI with prompt, temperature, top-k, interactive mode
│
├── visualize.py          ← The X-Ray (200 lines)
│   └── Heatmaps: single head, all heads, layer evolution
│
├── demo.py               ← The Showcase (200 lines)
│   └── Streamlit: generation + attention visualization
│
├── data/
│   └── input.txt         ← 1.1MB Shakespeare text
│
└── gpt_model.pt          ← Trained checkpoint (created after training)
```

---

## Complete Data Flow (The Final Picture)

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING                                  │
│                                                                  │
│  "ROMEO:" → [17,24,22,14,24,1]                                  │
│           → token_embed + pos_embed    (1, 6, 256)              │
│           → 6× TransformerBlock                                 │
│              norm→attn(+x)→norm→ffn(+x)                         │
│           → ln_final → lm_head          (1, 6, 65)              │
│           → CrossEntropyLoss(predicted, actual_next)            │
│           → loss.backward() → optimizer.step()                  │
│           → Repeat 10,000 times                                 │
│           → Loss: 4.17 → 1.08                                   │
│           → Save gpt_model.pt                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      GENERATION                                  │
│                                                                  │
│  "ROMEO:" → [17,24,22,14,24,1]                                  │
│           → model(x) → 65 logits for next char                  │
│           → / temperature → top-k filter → softmax              │
│           → sample → append → repeat 200 times                  │
│           → "ROMEO: I pray thee, gentle Mercutio..."            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Common Questions & Debugging

### "My loss is NaN!"
→ Learning rate too high. Try `--lr 1e-4`. Or your input has characters not in the vocabulary.

### "The model only generates the same character over and over!"
→ Temperature too low. Try `-t 1.0` or higher. Or the model didn't train long enough.

### "Training is too slow!"
→ Check you're using MPS (Apple Silicon) or CUDA. CPU training is ~10x slower.

### "The generated text doesn't look like Shakespeare!"
→ Train for more steps. 10K steps is minimum for coherent output. Try 20K for better quality.

### "What do B, T, C stand for?"
→ **B**atch (32), **T**ime/sequence (256), **C**hannels/embedding (256). Your compass through the code.

### "How do I make the model better?"
| Change | Effect |
|--------|--------|
| `embed_dim: 512` | 2× smarter, 4× slower |
| `num_layers: 12` | Deeper reasoning |
| `num_heads: 16` | More patterns detected |
| `block_size: 512` | Longer context memory |
| `max_steps: 20000` | More training time |

### "Can I train on my own text?"
Yes! Replace `data/input.txt` with any text file. The model automatically learns the vocabulary.

---

## What You Now Understand

| Concept | You can explain it to someone else |
|---------|-----------------------------------|
| Attention | Q compares to K, softmax, weighted V |
| Multi-head | 8 parallel attention patterns |
| Causal mask | Triangular matrix → no future peeking |
| Residual connection | `x = x + f(x)` → enables deep networks |
| LayerNorm | Normalize then scale/shift |
| FeedForward | Expand → filter → compress |
| Position embedding | How the model knows word order |
| Weight tying | Same weights for input and output |
| AdamW | Momentum + velocity + weight decay |
| Cosine LR | Fast learning then fine-tuning |
| Temperature | Controls creativity in generation |
| Top-k | Filters out unlikely tokens |
| Cross-entropy loss | Measures prediction accuracy |

---

## Next Steps After This Project

1. **Scale up**: Increase `embed_dim`, add more layers, train longer
2. **Switch to subword tokens**: Use `tiktoken` instead of characters
3. **Add RoPE**: Replace learned positions with rotary position encoding
4. **Train on more data**: Wikipedia, books, code
5. **Read the papers**: "Attention Is All You Need" (2017), GPT-2 paper (2019)

---

*Built with ❤️ for learners. If you made it this far, you now understand the architecture behind ChatGPT, GPT-4, and every modern language model.*
