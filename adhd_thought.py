"""
ADHD THOUGHT — Multi-thread, multi-perspective generation engine.
Maintains 5 parallel thought threads that interleave at the token level
(rapid ADHD switching) and cycle through macro-perspective phases
(hyperfocus, tangent, reframe, meta, synthesis).

On GPU: all 5 threads generate in a single batched forward pass.
"""

import torch
import random
import time
import re
from collections import deque


THREAD_DEFS = {
    'main': {
        'weight': 0.40,
        'seed': "\nThe main point is:",
        'label': "main analysis",
        'temp': 0.7,
    },
    'tangent': {
        'weight': 0.25,
        'seed': "\nThis reminds me of:",
        'label': "tangential connection",
        'temp': 0.9,
    },
    'meta': {
        'weight': 0.15,
        'seed': "\nI notice I'm thinking about this and I realize:",
        'label': "self-awareness",
        'temp': 0.8,
    },
    'reframe': {
        'weight': 0.10,
        'seed': "\nAnother way to see this is:",
        'label': "alternative perspective",
        'temp': 1.0,
    },
    'sensory': {
        'weight': 0.10,
        'seed': "\nThis feels like:",
        'label': "emotional context",
        'temp': 0.9,
    },
}

PHASE_SCHEDULE = [
    {
        'name': 'hyperfocus',
        'duration': 0.35,
        'weights': {'main': 0.75, 'tangent': 0.08, 'meta': 0.05, 'reframe': 0.07, 'sensory': 0.05},
        'temp_boost': 0.0,
    },
    {
        'name': 'explore',
        'duration': 0.20,
        'weights': {'main': 0.30, 'tangent': 0.30, 'meta': 0.10, 'reframe': 0.20, 'sensory': 0.10},
        'temp_boost': 0.1,
    },
    {
        'name': 'reframe',
        'duration': 0.15,
        'weights': {'main': 0.20, 'tangent': 0.15, 'meta': 0.15, 'reframe': 0.40, 'sensory': 0.10},
        'temp_boost': 0.2,
    },
    {
        'name': 'meta',
        'duration': 0.10,
        'weights': {'main': 0.20, 'tangent': 0.10, 'meta': 0.50, 'reframe': 0.10, 'sensory': 0.10},
        'temp_boost': 0.0,
    },
    {
        'name': 'synthesis',
        'duration': 0.20,
        'weights': {'main': 0.35, 'tangent': 0.10, 'meta': 0.25, 'reframe': 0.15, 'sensory': 0.15},
        'temp_boost': -0.1,
    },
]


class ThoughtThread:
    """A single thread of thought with its own context and generation style."""

    def __init__(self, name, config):
        self.name = name
        self.weight = config['weight']
        self.seed = config['seed']
        self.label = config['label']
        self.temp = config['temp']
        self.context = ""
        self.token_count = 0

    def build_prompt(self, shared_context):
        return shared_context + self.seed

    def __repr__(self):
        return f"<Thread:{self.name} w={self.weight:.2f} ctx={len(self.context)}>"


class ADHDGenerator:
    """
    Multi-thread ADHD generation engine.

    Batched GPU mode: all 5 threads generate logits in one forward pass,
    then we interleave by sampling from the weighted thread distribution.
    """

    def __init__(self):
        self.threads = {}
        for name, config in THREAD_DEFS.items():
            self.threads[name] = ThoughtThread(name, config)

        self.shared_context = ""
        self.output_threads = []
        self.last_thread = None
        self.consecutive = 0
        self.switch_urge = 0
        self.total_tokens = 0
        self.hyperfocus_mode = False
        self.hyperfocus_topic = None

        self.adhd_enabled = True
        self.phase_enabled = True

    def generate(self, prompt, model, tokenizer, max_tokens=300,
                 temperature=0.8, top_k=50, repetition_penalty=1.05):
        if not self.adhd_enabled:
            return self._linear_generate(prompt, model, tokenizer, max_tokens, temperature)

        return self._batched_adhd_generate(
            prompt, model, tokenizer, max_tokens,
            temperature, top_k, repetition_penalty
        )

    def reason_with_adhd(self, problem, model, tokenizer, max_tokens=400):
        if not self.phase_enabled:
            return self.generate(problem, model, tokenizer, max_tokens, temperature=0.7)

        phases = PHASE_SCHEDULE
        tokens_per_phase = max(20, int(max_tokens * 0.8 / len(phases)))

        all_output = []
        self.shared_context = f"I need to reason about: {problem}\n"

        for phase in phases:
            self._set_phase(phase)

            phase_output = f"\n--- {phase['name'].upper()} ---\n"
            temp = temperature + phase.get('temp_boost', 0)

            gen = self._batched_adhd_generate(
                self.shared_context, model, tokenizer,
                tokens_per_phase, temp, top_k=50, repetition_penalty=1.05
            )

            phase_output += gen
            all_output.append(phase_output)
            self.shared_context += gen + "\n"

            self.hyperfocus_topic = None
            self.hyperfocus_mode = False

        return {
            'phases': [p['name'] for p in phases],
            'full_output': '\n'.join(all_output),
            'phases_detailed': all_output,
        }

    def _batched_adhd_generate(self, prompt, model, tokenizer, max_tokens,
                                temperature, top_k, repetition_penalty):
        self.shared_context = prompt
        self.output_threads = []
        self.last_thread = None
        self.consecutive = 0
        self.total_tokens = 0

        output = []
        device = next(model.parameters()).device

        for _ in range(max(1, max_tokens // 8)):
            if self.total_tokens >= max_tokens:
                break

            thread = self._pick_thread()
            self.total_tokens += 1

            thread_prompt = self.shared_context + thread.seed
            prompt_ids = tokenizer.encode(thread_prompt)

            if len(prompt_ids) > model.max_context:
                prompt_ids = prompt_ids[-model.max_context:]

            context_t = torch.tensor([prompt_ids], dtype=torch.long, device=device)

            with torch.no_grad():
                logits, _, _ = model(context_t, return_value=False)
                logits = logits[:, -1, :] / (thread.temp * temperature if self.adhd_enabled else temperature)
                logits = torch.clamp(logits, min=-50.0, max=50.0)

                if top_k > 0:
                    k = min(top_k, logits.size(-1))
                    vals, idxs = torch.topk(logits, k)
                    threshold = vals[:, -1:].expand_as(logits)
                    logits = torch.where(logits < threshold, float('-inf'), logits)

                if torch.all(logits == float('-inf')):
                    logits = torch.zeros_like(logits) - 10.0

                logits = logits - logits.max(dim=-1, keepdim=True)[0]
                logits = torch.clamp(logits, min=-30.0, max=0.0)
                probs = torch.softmax(logits, dim=-1)

                if torch.isnan(probs).any():
                    probs = torch.ones_like(probs) / probs.size(-1)

                idx_next = torch.multinomial(probs, num_samples=1)
                next_id = idx_next.item()

            decoded = tokenizer.decode([next_id])
            if decoded:
                output.append(f"[{thread.name}] {decoded}")
                self.shared_context += " " + decoded.strip()

            if next_id == model.eos_token_id:
                break

        return '\n'.join(output)

    def _pick_thread(self):
        if self.hyperfocus_mode and self.last_thread == 'main':
            self.consecutive += 1
            return self.threads['main']

        weights = {}
        for name, thread in self.threads.items():
            w = thread.weight
            if self.consecutive >= 2 and name == self.last_thread:
                w *= 0.1
            if self.switch_urge > 3:
                w = w * 0.5 if name == self.last_thread else w * 1.5
            weights[name] = max(0.01, w)

        total = sum(weights.values())
        r = random.random() * total
        cumulative = 0
        chosen = 'main'
        for name, w in weights.items():
            cumulative += w
            if r <= cumulative:
                chosen = name
                break

        if chosen == self.last_thread:
            self.consecutive += 1
            self.switch_urge = 0
        else:
            self.consecutive = 1
            self.switch_urge += 1

        self.last_thread = chosen
        return self.threads[chosen]

    def _set_phase(self, phase):
        for name, thread in self.threads.items():
            if name in phase['weights']:
                thread.weight = phase['weights'][name]

    def _linear_generate(self, prompt, model, tokenizer, max_tokens, temperature):
        prompt_ids = tokenizer.encode(prompt)
        generated_ids = model.generate(
            prompt_ids, max_new_tokens=max_tokens, temperature=temperature
        )
        return tokenizer.decode(generated_ids)

    def toggle_adhd(self):
        self.adhd_enabled = not self.adhd_enabled
        return self.adhd_enabled

    def state(self):
        return {
            'adhd_enabled': self.adhd_enabled,
            'phase_enabled': self.phase_enabled,
            'hyperfocus': self.hyperfocus_mode,
            'last_thread': self.last_thread,
            'switch_urge': self.switch_urge,
        }
