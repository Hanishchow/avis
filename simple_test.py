import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# Simple test
x = torch.randn(3, 3)
print("Random tensor:")
print(x)