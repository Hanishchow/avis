"""
Massive training pipeline: Nemotron-Cascade-2-SFT-Data streaming, checkpointing, resume.
Usage:
  python train_massive.py              # fresh start
  python train_massive.py --resume      # resume from latest checkpoint
"""
import sys, os, time, glob, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from datasets import load_dataset
from collections import deque
import torch
from biologic_v2 import create_model, tokenizer

# Load HF token if saved
if os.path.exists('.hf_token'):
    with open('.hf_token') as f:
        tok = f.read().strip()
        if tok:
            os.environ['HF_TOKEN'] = tok
            print(f"HF_TOKEN loaded from .hf_token", flush=True)

CKPT_DIR = "checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)
resume = '--resume' in sys.argv
fast_mode = '--fast' in sys.argv
if fast_mode:
    print("FAST MODE: larger chunks, skipping Hebbian on every other step", flush=True)

chunk_size = 96 if fast_mode else 48
stride = 48 if fast_mode else 24

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", flush=True)

# --- Find latest checkpoint ---
latest_ckpt = None
if resume:
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "step_*.pt")),
                    key=lambda x: int(x.replace('_final','').split('_')[-1].split('.')[0]))
    if ckpts:
        latest_ckpt = ckpts[-1]
        print(f"Resuming from: {latest_ckpt}", flush=True)
    else:
        print("No checkpoints found, starting fresh.", flush=True)

if latest_ckpt:
    # Load checkpoint
    ckpt = torch.load(latest_ckpt, map_location=device)
    model = create_model(tokenizer_ref=tokenizer, do_seed_learning=False, auto_scale=True)
    model.load_state_dict(ckpt['model_state'])
    model.train()
    for mod in model.modules():
        if hasattr(mod, 'use_meta') and mod.use_meta:
            mod.meta_base.data.fill_(0.01)
            mod.meta_surprise_scale.data.fill_(1.0)
            mod.meta_gate.data.zero_()
            mod.meta_decay.data.fill_(-3.0)
            mod.meta_act_scale.data.fill_(0.1)
    total_steps = ckpt['total_steps']
    model.total_experience = ckpt.get('experience', 0)
    print(f"Resumed at step {total_steps}, experience {model.total_experience}", flush=True)
else:
    print("Creating model (no seed learning — dataset will teach it)...", flush=True)
    model = create_model(tokenizer_ref=tokenizer, do_seed_learning=False, auto_scale=True)
    model.train()
    total_steps = 0
    # Init meta-params to safe defaults
    for mod in model.modules():
        if hasattr(mod, 'use_meta') and mod.use_meta:
            mod.meta_base.data.fill_(0.01)
            mod.meta_surprise_scale.data.fill_(1.0)
            mod.meta_gate.data.zero_()
            mod.meta_decay.data.fill_(-3.0)
            mod.meta_act_scale.data.fill_(0.1)

print(f"Model: {sum(p.numel() for p in model.parameters()):,} params", flush=True)

# --- Training loop ---
configs = ["chat", "conversational_agent", "instruction_following"]
nan_count = 0
step_times = []
recent_losses = deque(maxlen=200)
checkpoint_interval = 500
report_interval = 50
last_ckpt_step = total_steps

try:
    while True:
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
                        texts.extend([x.get('text', '') for x in c if isinstance(x, dict) and x.get('text')])
                text = ' '.join(texts)
                if len(text) < 20:
                    continue
                encoded = tokenizer.encode(text)
                if len(encoded) < chunk_size + 2:
                    continue
                for sp in range(0, len(encoded) - chunk_size - 1, stride):
                    ck = encoded[sp:sp + chunk_size]
                    tg = encoded[sp + 1:sp + chunk_size + 1]
                    if len(ck) != len(tg) or len(ck) < 2:
                        continue
                    t0 = time.time()
                    result = model.learn_from_interaction(ck, tg, value_label=0.5, task_type=cfg)
                    step_times.append(time.time() - t0)
                    if result['loss'] is None or np.isnan(result['loss']):
                        nan_count += 1
                    else:
                        total_steps += 1
                        recent_losses.append(result['loss'])
                        # Checkpoint per-step
                        if total_steps - last_ckpt_step >= checkpoint_interval:
                            ckpt_path = os.path.join(CKPT_DIR, f"step_{total_steps}.pt")
                            torch.save({
                                'model_state': model.state_dict(),
                                'total_steps': total_steps,
                                'experience': model.total_experience,
                                'nan_count': nan_count,
                            }, ckpt_path)
                            print(f"  [CKPT] step={total_steps} saved (loss={result['loss']:.4f})", flush=True)
                            last_ckpt_step = total_steps
                            # Also print summary at every checkpoint
                            avg_t = np.mean(step_times[-100:]) if step_times else 0
                            sps = 1.0 / avg_t if avg_t > 0 else 0
                            avg_loss = np.nanmean(list(recent_losses)[-100:]) if recent_losses else 0
                            print(
                                f"  [{cfg}] steps={total_steps} | loss={avg_loss:.4f} | "
                                f"{sps:.1f}step/s | nan={nan_count} | exp={model.total_experience}",
                                flush=True
                            )

            print(f"  [{cfg}] Config done: {ex_count} examples", flush=True)

except KeyboardInterrupt:
    print("\n\nInterrupted — saving final checkpoint...", flush=True)

ckpt_path = os.path.join(CKPT_DIR, f"step_{total_steps}_final.pt")
torch.save({
    'model_state': model.state_dict(),
    'total_steps': total_steps,
    'experience': model.total_experience,
    'nan_count': nan_count,
}, ckpt_path)
print(f"Saved {ckpt_path}", flush=True)

print(f"\n=== TRAINING SUMMARY ===", flush=True)
print(f"Total steps: {total_steps}, NaN count: {nan_count}", flush=True)
print(f"Total experience: {model.total_experience}", flush=True)

print("\n=== GENERATION TEST ===", flush=True)
model.eval()
prompt = tokenizer.encode("What is consciousness?")
gen = model.generate(prompt, max_new_tokens=80, temperature=0.4, top_k=40, repetition_penalty=1.1)
decoded = tokenizer.decode(gen)
print(f"  {decoded[:500]}", flush=True)
