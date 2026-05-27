"""Test the trained model from the latest checkpoint"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
from biologic_v2 import create_model, tokenizer
import glob

# Find latest checkpoint
ckpts = sorted(glob.glob("checkpoints/step_*.pt"),
               key=lambda x: int(x.replace('_final','').split('_')[-1].split('.')[0]))
latest = ckpts[-1]
print(f"Loading checkpoint: {latest}", flush=True)

# Create model and load weights
model = create_model(tokenizer_ref=tokenizer, do_seed_learning=False, auto_scale=True)
ckpt = torch.load(latest, map_location='cuda')
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"Loaded: step={ckpt['total_steps']}, experience={ckpt['experience']}", flush=True)

# Test prompts
prompts = [
    "What is consciousness?",
    "Explain how neural networks work.",
    "What is the meaning of life?",
    "How does gravity work?",
    "Write a short poem about AI.",
]

for prompt in prompts:
    encoded = tokenizer.encode(prompt)
    gen = model.generate(encoded, max_new_tokens=80, temperature=0.4, top_k=40, repetition_penalty=1.1)
    decoded = tokenizer.decode(gen)
    print(f"\nQ: {prompt}", flush=True)
    print(f"A: {decoded[:200]}", flush=True)
