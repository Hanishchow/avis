import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

# Step 3: Creating our character-level dataset
class CharacterDataset:
    def __init__(self, text, seq_length):
        self.seq_length = seq_length
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        
        # Convert text to indices
        self.data = [self.char_to_idx[c] for c in text]
        
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        # Get sequence and target
        seq = self.data[idx:idx + self.seq_length]
        target = self.data[idx + 1:idx + self.seq_length + 1]
        return torch.tensor(seq), torch.tensor(target)

# Step 4: Building the transformer model components
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads"
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Linear transformations
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out(attn_output)
        
        return output

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, forward_expansion, dropout):
        super().__init__()
        self.attention = SelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Using ReLU activation as requested
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion * embed_dim),
            nn.ReLU(),  # ReLU activation
            nn.Linear(forward_expansion * embed_dim, embed_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention with residual connection
        attention = self.attention(x)
        x = self.norm1(attention + x)
        x = self.dropout(x)
        
        # Feed forward with residual connection
        forward = self.feed_forward(x)
        x = self.norm2(forward + x)
        x = self.dropout(x)
        
        return x

# Step 5: Creating our complete mini LLM
class MiniLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, 
                 forward_expansion, seq_length, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_length = seq_length
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # Positional embedding
        self.position_embedding = nn.Embedding(seq_length, embed_dim)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, forward_expansion, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm and output projection
        self.norm = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len = x.size()
        
        # Create positions
        positions = torch.arange(0, seq_len).expand(batch_size, seq_len).to(x.device)
        
        # Embed tokens and positions
        tokens = self.token_embedding(x)
        positions = self.position_embedding(positions)
        x = self.dropout(tokens + positions)
        
        # Pass through transformer blocks
        for layer in self.layers:
            x = layer(x)
        
        # Final norm and projection
        x = self.norm(x)
        logits = self.output(x)
        
        return logits
    
    def generate(self, start_string, predict_len, temperature=0.8):
        self.eval()
        # Convert start string to indices
        indices = [dataset.char_to_idx[c] for c in start_string]
        input_seq = torch.tensor(indices).unsqueeze(0).long()
        
        generated = []
        
        with torch.no_grad():
            for _ in range(predict_len):
                # Get predictions
                logits = self.forward(input_seq)
                # Get last token predictions
                logits = logits[:, -1, :] / temperature
                # Apply softmax to get probabilities
                probs = F.softmax(logits, dim=-1)
                # Sample from distribution
                idx_next = torch.multinomial(probs, num_samples=1)
                
                # Append predicted index
                generated.append(idx_next.item())
                # Update input sequence
                input_seq = torch.cat((input_seq, idx_next), dim=1)
                # Keep only last seq_length tokens
                if input_seq.size(1) > self.seq_length:
                    input_seq = input_seq[:, -self.seq_length:]
        
        # Convert indices back to characters
        generated_chars = [dataset.idx_to_char[i] for i in generated]
        return start_string + ''.join(generated_chars)

# Step 6: Training function
def train_model(model, dataset, epochs=200, batch_size=64, learning_rate=0.005):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for seq, target in dataloader:
            optimizer.zero_grad()
            logits = model(seq)
            # Reshape for loss calculation
            loss = criterion(logits.view(-1, dataset.vocab_size), target.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')
            
            # Generate sample text
            sample_texts = [
                "The quick brown fox",
                "To be or not to",
                "All that glitters",
                "Where there is a",
                "We hold these truths"
            ]
            for start in sample_texts[:2]:  # Just show 2 samples to save space
                sample_text = model.generate(start, 100, temperature=0.7)
                print(f'Sample: "{sample_text[:80]}..."')
            print()

# Step 7: Main execution
if __name__ == "__main__":
    # Load and prepare data
    with open('training_data.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Text length: {len(text)} characters")
    print(f"Unique characters: {len(set(text))}")
    
    # Hyperparameters - increased for better language modeling
    SEQ_LENGTH = 30
    EMBED_DIM = 128
    NUM_HEADS = 8
    NUM_LAYERS = 3
    FORWARD_EXPANSION = 4
    DROPOUT = 0.1
    
    # Create dataset
    dataset = CharacterDataset(text, SEQ_LENGTH)
    print(f"Vocabulary size: {dataset.vocab_size}")
    # Safely print characters for display
    chars_str = ''.join(c for c in dataset.chars if c.isprintable() or c in [' ', '\n', '\t', ',', '.', '!', '?', ';', ':'])
    print(f"Characters: {repr(chars_str)}")
    
    # Create model
    model = MiniLLM(
        vocab_size=dataset.vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        forward_expansion=FORWARD_EXPANSION,
        seq_length=SEQ_LENGTH,
        dropout=DROPOUT
    )
    
    print(f"\nModel architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Train model
    print("\nStarting training...")
    train_model(model, dataset, epochs=200, batch_size=64, learning_rate=0.005)
    
    # Generate final samples
    print("\nFinal generation samples:")
    test_starts = [
        "The quick brown fox",
        "To be or not to",
        "All that glitters",
        "Where there is a",
        "We hold these truths",
        "It is a truth",
        "It was the best",
        "To infinity and",
        "May the Force",
        "A journey of"
    ]
    
    for start in test_starts:
        generated = model.generate(start, 150, temperature=0.7)
        print(f'Start: "{start}"')
        print(f'Generated: "{generated}"\n')