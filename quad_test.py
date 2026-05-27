import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

# Minimal character-level transformer with ReLU
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, num_heads=4, num_layers=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(64, embed_dim)  # max seq length 64
        
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim*4,
                activation='relu',  # Using ReLU as requested
                batch_first=True,
                dropout=0.1
            ) for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, idx):
        b, t = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.pos_embedding(torch.arange(t, device=idx.device))
        x = tok_emb + pos_emb
        
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

def train_simple_model():
    # Use quotes from literature for better English
    text = """To be or not to be, that is the question.
All that glitters is not gold.
Where there is a will, there is a way.
We hold these truths to be self-evident.
It was the best of times, it was the worst of times.
To infinity and beyond.
May the Force be with you.
A journey of a thousand miles begins with a single step."""
    
    # Create character mappings
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    vocab_size = len(chars)
    
    # Encode text
    data = [stoi[c] for c in text]
    
    # Create training examples
    block_size = 16
    torch.manual_seed(1337)
    
    def get_batch():
        ix = torch.randint(len(data) - block_size, (32,))
        x = torch.stack([torch.tensor(data[i:i+block_size]) for i in ix])
        y = torch.stack([torch.tensor(data[i+1:i+block_size+1]) for i in ix])
        return x, y
    
    # Initialize model
    model = SimpleTransformer(vocab_size=vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # Train for a few iterations
    print(f"Vocabulary size: {vocab_size}")
    print(f"Training on {len(text)} characters")
    print("Characters:", ''.join(chars))
    print("\nTraining...")
    
    for iter in range(500):
        xb, yb = get_batch()
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if iter % 50 == 0:
            print(f"Step {iter}: loss {loss.item():.4f}")
    
    # Generate text
    print("\nGenerating text:")
    model.eval()
    start = "To be or not"
    context = torch.tensor([stoi[c] for c in start], dtype=torch.long).unsqueeze(0)
    
    generated = []
    for _ in range(100):
        # Get predictions
        logits = model(context)
        # Focus on last time step
        logits = logits[:, -1, :] / 0.8  # temperature
        probs = F.softmax(logits, dim=-1)
        # Sample from distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        # Append to context
        context = torch.cat((context, idx_next), dim=1)
        # Keep only last block_size tokens
        if context.size(1) > block_size:
            context = context[:, -block_size:]
        generated.append(idx_next.item())
    
    # Decode generated text
    generated_text = start + ''.join([itos[i] for i in generated])
    print(f'Start: "{start}"')
    print(f'Generated: "{generated_text}"')

if __name__ == "__main__":
    train_simple_model()