import torch
import torch.nn as nn
import torch.nn.functional as F

# Super simple character-level RNN for testing
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden):
        x = self.embedding(x)
        output, hidden = self.rnn(x, hidden)
        output = self.fc(output)
        return output, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.rnn.hidden_size)

# Very small test
if __name__ == "__main__":
    # Simple vocabulary
    chars = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?-"
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    # Training text
    text = "To be or not to be, that is the question."
    # Convert to indices
    data = [char_to_idx[c] for c in text]
    
    # Hyperparameters
    seq_length = 10
    embedding_dim = 8
    hidden_dim = 16
    
    # Create model
    model = CharRNN(vocab_size, embedding_dim, hidden_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Create training data
    inputs = []
    targets = []
    for i in range(len(data) - seq_length):
        inputs.append(data[i:i+seq_length])
        targets.append(data[i+1:i+seq_length+1])
    
    inputs = torch.tensor(inputs)
    targets = torch.tensor(targets)
    
    # Train for a few epochs
    print("Training simple RNN...")
    for epoch in range(50):
        optimizer.zero_grad()
        hidden = model.init_hidden(inputs.size(0))
        output, _ = model(inputs, hidden)
        loss = criterion(output.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/50], Loss: {loss.item():.4f}')
    
    # Test generation
    model.eval()
    start_string = "To be"
    input_seq = [char_to_idx[c] for c in start_string]
    input_tensor = torch.tensor(input_seq).unsqueeze(0)
    
    generated = []
    hidden = model.init_hidden(1)
    
    with torch.no_grad():
        for _ in range(20):
            output, hidden = model(input_tensor, hidden)
            probs = F.softmax(output[:, -1, :], dim=-1)
            next_char_idx = torch.multinomial(probs, num_samples=1)
            generated.append(next_char_idx.item())
            input_tensor = torch.cat((input_tensor, next_char_idx), dim=1)
            if input_tensor.size(1) > seq_length:
                input_tensor = input_tensor[:, -seq_length:]
    
    generated_chars = [idx_to_char[i] for i in generated]
    result = start_string + ''.join(generated_chars)
    print(f'\nGenerated text: "{result}"')
    print("Simple RNN test completed!")