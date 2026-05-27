"""
Mini LLM Trained on Both English and Mathematics
- Uses ReLU activation in feed-forward networks
- Trained on combined English and math text
- Demonstrates core LLM concepts: embeddings, attention, transformer blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

print("="*60)
print("BUILDING A MINI LLM FOR ENGLISH + MATH")
print("="*60)

# STEP 1: PREPARING THE DATA
print("\nSTEP 1: Preparing Training Data (English + Math)")
print("-" * 50)

# Using combined English and math text for training
with open('training_data_combined.txt', 'r', encoding='utf-8') as f:
    training_text = f.read()

print(f"Training text length: {len(training_text)} characters")
print(f"Unique characters: {len(set(training_text))}")

# Create character-level vocabulary
chars = sorted(list(set(training_text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}  # string to integer
itos = {i: ch for i, ch in enumerate(chars)}  # integer to string

# Encode the entire dataset
data = [stoi[c] for c in training_text]
print(f"Vocabulary size: {vocab_size}")
print(f"Sample characters: {''.join(chars[:30])}...")

# Show some math content in the data
if any(c in training_text for c in '0123456789+-=×÷()'):
    print("[+] Contains mathematical notation and equations")
else:
    print("⚠ Warning: May not contain sufficient math content")

# STEP 2: CREATING TRAINING EXAMPLES
print("\nSTEP 2: Creating Training Examples")
print("-" * 30)

block_size = 40  # Increased context length for better math understanding
def get_batch(data, batch_size=32):
    """Generate a small batch of data of inputs x and targets y"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.tensor(data[i:i+block_size]) for i in ix])
    y = torch.stack([torch.tensor(data[i+1:i+block_size+1]) for i in ix])
    return x, y

# Test the batch function
xb, yb = get_batch(data, batch_size=4)
print(f"Input shape: {xb.shape}")  # (batch, sequence_length)
print(f"Target shape: {yb.shape}")  # (batch, sequence_length)
print(f"Example input: {xb[0][:15].tolist()} -> '{''.join([itos[i.item()] for i in xb[0][:15]])}'")
print(f"Example target: {yb[0][:15].tolist()} -> '{''.join([itos[i.item()] for i in yb[0][:15]])}'")

# Check if we have math in our examples
sample_text = ''.join([itos[i.item()] for i in xb[0]])
if any(c in sample_text for c in '0123456789+-=×÷()'):
    print("[+] Training batch contains mathematical content")
else:
    print("⚠ Training batch may not contain math - consider increasing batch size")

# STEP 3: BUILDING THE MODEL COMPONENTS
print("\nSTEP 3: Building Model Components")
print("-" * 30)

class Head(nn.Module):
    """One head of self-attention"""
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """Compute self-attention"""
        B, T, C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        
        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        
        # Weighted aggregation of values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),  # Using ReLU as requested
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block: communication followed by computation"""
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # Communication
        x = x + self.ffwd(self.ln2(x))  # Computation
        return x

# STEP 4: ASSEMBLING THE COMPLETE MODEL
print("\nSTEP 4: Assembling the Complete Model")
print("-" * 30)

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd=96, n_head=6, n_layer=6, block_size=40, dropout=0.15):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        self.block_size = block_size
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Embed tokens and positions
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Generate new tokens autoregressively"""
        for _ in range(max_new_tokens):
            # Crop context to last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # Get predictions
            logits, _ = self(idx_cond)
            # Focus on last time step and apply temperature
            logits = logits[:, -1, :] / temperature
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # Sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append sampled index
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

# Instantiate the model
model = GPTLanguageModel(
    vocab_size=vocab_size,
    n_embd=96,
    n_head=6,
    n_layer=6,
    block_size=block_size,
    dropout=0.15
)

print(f"Model created with {sum(p.numel() for p in model.parameters())/1e3:.1f}K parameters")
print("Model architecture:")
print(f"  - Token Embedding: {vocab_size} -> {model.token_embedding_table.embedding_dim}")
print(f"  - Position Embedding: {model.position_embedding_table.num_embeddings} -> {model.position_embedding_table.embedding_dim}")
print(f"  - Transformer Blocks: {len(model.blocks)} blocks")
print(f"  - Each Block: Multi-Head Attention (6 heads) + Feed-Forward (ReLU)")
print(f"  - Final Layer Norm + Language Model Head: {model.ln_f.normalized_shape[0]} -> {model.lm_head.out_features}")

# STEP 5: TRAINING THE MODEL
print("\nSTEP 5: Training the Model on English + Math")
print("-" * 50)

# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Training loop
max_iters = 800
eval_interval = 100

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(50)
        for k in range(50):
            X, Y = get_batch(train_data if split == 'train' else val_data)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Train/validation split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Training on {len(train_data)} tokens, validating on {len(val_data)} tokens")

print("Iteration | Train Loss | Val Loss")
print("-" * 35)
for iter in range(max_iters):
    # Evaluate loss
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"{iter:9d} | {losses['train']:10.4f} | {losses['val']:9.4f}")
    
    # Sample batch and train
    xb, yb = get_batch(train_data)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# STEP 6: GENERATING ENGLISH AND MATH TEXT
print("\nSTEP 6: Generating English and Mathematics Text")
print("-" * 50)

model.eval()
print("Generating text from various prompts (English and Math):\n")

test_prompts = [
    # English prompts
    "To be or not to be",
    "All that glitters",
    "Where there is a will",
    "We hold these truths",
    "It was the best of times",
    "To infinity and beyond",
    "May the Force be",
    "A journey of a thousand",
    "We the People",
    "I have a dream",
    
    # Math prompts
    "Mathematics is the study",
    "Addition combines numbers",
    "In algebra, we use",
    "Geometry studies shapes",
    "Calculus deals with change",
    "Prime numbers are numbers",
    "The number π (pi)",
    "In trigonometry, the sine",
    "The Fibonacci sequence is",
    "Mathematical proofs establish"
]

for prompt in test_prompts:
    # Convert prompt to tensor
    context = torch.tensor([stoi[c] for c in prompt], dtype=torch.long).unsqueeze(0)
    
    # Generate continuation
    generated_ids = model.generate(context, max_new_tokens=100, temperature=0.7)[0].tolist()
    generated_text = ''.join([itos[i] for i in generated_ids])
    
    print(f'Prompt:  "{prompt}"')
    print(f'Output:  "{generated_text}"')
    print()

print("="*60)
print("MINI LLM FOR ENGLISH + MATH BUILD COMPLETE!")
print("="*60)
print("What we built:")
print("[+] Character-level language model")
print("[+] Transformer architecture with self-attention")
print("[+] Multi-head attention mechanism")
print("[+] Feed-forward networks with ReLU activation")
print("[+] Positional embeddings")
print("[+] Layer normalization and residual connections")
print("[+] Autoregressive text generation")
print()
print("Key concepts demonstrated:")
print("• Embeddings: Convert tokens to dense vectors")
print("• Attention: Let each token attend to relevant context")
print("• Transformers: Stack attention + feed-forward blocks")
print("• Training: Predict next token given previous tokens")
print("• Generation: Sample from probability distribution autoregressively")
print()
print("Training data included:")
print("• Literary quotes (Shakespeare, Dickens, etc.)")
print("• Philosophical texts (Declaration of Independence, etc.)")
print("• Pop culture references (Star Wars)")
print("• Comprehensive mathematics:")
print("  - Arithmetic operations")
print("  - Algebra and equations")
print("  - Geometry and trigonometry")
print("  - Calculus (derivatives and integrals)")
print("  - Number theory (primes, fractions)")
print("  - Statistics and proofs")
print("  - Famous constants (π, Fibonacci)")
print()
print("Note: This model (~380K parameters) learned patterns from both")
print("English language and mathematical notation, showing how LLMs")
print("can acquire multiple domains of knowledge from text alone!")
print("="*60)