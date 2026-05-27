import torch
import torch.nn as nn
import torch.nn.functional as F

print("Testing basic PyTorch functionality...")

# Simple test
x = torch.randn(2, 3)
print("Input:", x)

# Simple linear layer
linear = nn.Linear(3, 4)
output = linear(x)
print("Output shape:", output.shape)
print("Output:", output)

print("Test completed successfully!")