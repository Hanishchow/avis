"""
Train Biologic on the full Nemotron-Cascade-2-SFT-Data.
Run: pip install datasets  (first time only)
Then: python -u train_on_dataset.py
"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
from biologic_v2 import create_model, tokenizer
from datasets import load_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", flush=True)

# Fast model creation — skip seed learning, we train on the real dataset
model = create_model(tokenizer_ref=tokenizer, do_seed_learning=False, auto_scale=True)
model.train()
print(f"Model: {sum(p.numel() for p in model.parameters()):,} params", flush=True)

# Reset meta-params to safe defaults
for mod in model.modules():
    if hasattr(mod, 'use_meta') and mod.use_meta:
        mod.meta_base.data.fill_(0.01)
        mod.meta_surprise_scale.data.fill_(1.0)
        mod.meta_gate.data.zero_()
        mod.meta_decay.data.fill_(-3.0)
        mod.meta_act_scale.data.fill_(0.1)

configs = ["chat", "conversational_agent", "instruction_following"]
chunk_size = 48
stride = 24
total_steps = 0
nan_count = 0
step_times = []

for cfg in configs:
    print(f"\n=== Loading config: {cfg} ===", flush=True)
    ds = load_dataset("nvidia/Nemotron-Cascade-2-SFT-Data", cfg, split="train", streaming=True)
    ex_count = 0
    for example in ds:
        ex_count += 1
        msg = example.get('messages', [])
        texts = []
        for m in msg:
            c = m.get('content', '')
            if isinstance(c, str):
                texts.append(c)
            elif isinstance(c, list):
                texts.extend([x.get('text','') for x in c if isinstance(x, dict) and x.get('text')])
        text = ' '.join(texts)
        if len(text) < 20:
            continue
        encoded = tokenizer.encode(text)
        if len(encoded) < chunk_size + 2:
            continue
        for sp in range(0, len(encoded) - chunk_size - 1, stride):
            ck = encoded[sp:sp+chunk_size]
            tg = encoded[sp+1:sp+chunk_size+1]
            if len(ck) != len(tg) or len(ck) < 2:
                continue
            t0 = time.time()
            result = model.learn_from_interaction(ck, tg, value_label=0.5, task_type=cfg)
            step_times.append(time.time() - t0)
            if result['loss'] is None or np.isnan(result['loss']):
                nan_count += 1
            else:
                total_steps += 1
        if ex_count % 50 == 0:
            avg_t = np.mean(step_times[-100:]) if step_times else 0
            print(f"  [{cfg}] {ex_count} examples, {total_steps} steps, {avg_t*1000:.0f}ms/step, nan={nan_count}", flush=True)
    print(f"  [{cfg}] Done: {ex_count} examples", flush=True)

print(f"\n=== TRAINING COMPLETE ===", flush=True)
print(f"Total steps: {total_steps}, NaN: {nan_count}", flush=True)
print(f"Total experience: {model.total_experience}", flush=True)

model.consolidate_memory()

print("\n=== GENERATION TEST ===", flush=True)
model.eval()
prompt = tokenizer.encode("What is consciousness?")
gen = model.generate(prompt, max_new_tokens=80, temperature=0.4, top_k=40, repetition_penalty=1.1)
decoded = tokenizer.decode(gen)
print(f"  {decoded[:500]}", flush=True)
