"""
BIOLOGIC LLM V2 — BPE tokenization + 5000+ token sliding window context + web learning + reasoning.
GPU-accelerated: batch generation, human-like gestalt, scaled model for 6GB VRAM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import random
import json
from datetime import datetime
from collections import deque

from sliding_attention import ChunkedSlidingAttention
from tokenizer import BPETokenizer

print("=" * 60)
print("BIOLOGIC LLM V2 — INITIALIZING")
print("=" * 60)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[DEVICE] {DEVICE} {'(' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else ''}")

SEED_TEXTS = {
    "english": """
To be or not to be, that is the question. All that glitters is not gold.
Where there is a will, there is a way. We hold these truths to be self-evident.
It was the best of times, it was the worst of times.
The quick brown fox jumps over the lazy dog.
May the Force be with you. I think, therefore I am.
Language is the tool of thought. Words shape how we perceive reality.
Grammar gives structure to meaning. Every sentence carries intent and context.
The English language borrows words from Latin, Greek, French, and German.
Vocabulary grows through reading and conversation. Writing clarifies thinking.
Metaphor allows us to understand one thing in terms of another.
Storytelling is how humans make sense of the world.
Questions are more important than answers because they keep curiosity alive.
""",
    "math": """
Addition: 2 + 3 = 5. Subtraction: 5 - 2 = 3. Multiplication: 3 * 4 = 12.
Division: 12 / 3 = 4. In algebra, x + 5 = 10 means x = 5.
The quadratic formula: x = (-b +/- sqrt(b^2 - 4ac)) / (2a).
Pythagorean theorem: a^2 + b^2 = c^2.
Prime numbers: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29.
Pi = 3.14159. Fibonacci: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89.
Mathematics is the language of patterns. Calculus studies change and motion.
Integration finds area under curves. Differentiation finds instantaneous rate of change.
Linear algebra studies vector spaces and linear transformations.
Statistics describes data through mean, median, variance, and standard deviation.
Probability measures uncertainty on a scale from zero to one.
Set theory is the foundation of modern mathematics.
A function maps every input to exactly one output.
The number line extends infinitely in both directions.
Complex numbers combine real and imaginary parts.
Logarithms transform multiplication into addition.
""",
    "code": """
def factorial(n): return 1 if n <= 1 else n * factorial(n - 1)
def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)
def quicksort(arr): return arr if len(arr) <= 1 else quicksort([x for x in arr[1:] if x <= arr[0]]) + [arr[0]] + quicksort([x for x in arr[1:] if x > arr[0]])
class Stack: def __init__(self): self.items = []; def push(self, item): self.items.append(item); def pop(self): return self.items.pop()
A variable stores a value. A function encapsulates reusable logic.
Loops repeat operations. Conditionals branch execution based on truth.
Arrays store sequences of elements. Dictionaries map keys to values.
Recursion solves a problem by solving smaller instances of the same problem.
Time complexity measures how runtime grows with input size.
Space complexity measures how memory usage grows with input size.
Data structures organize information for efficient access and modification.
Linked lists connect nodes through pointers. Trees have a root and branches.
Hash tables provide constant-time lookup on average.
Graphs model relationships between entities.
Binary search finds an element in sorted data in logarithmic time.
Dynamic programming solves problems by combining solutions to subproblems.
An algorithm is a finite sequence of well-defined instructions.
The Python programming language emphasizes readability and simplicity.
Indentation defines blocks of code in Python.
""",
    "ethics": """
Knowledge should be used to help, not harm. Always verify information before trusting it.
Respect privacy and consent. Do not generate malicious code or instructions for harm.
Be honest about your limitations. Learn from mistakes and correct them.
Protect those who cannot protect themselves. Truth is more important than being right.
The purpose of intelligence is understanding, not control.
Ethics is the branch of philosophy concerned with right and wrong.
The golden rule states: treat others as you would like to be treated.
Deontology judges actions by their adherence to moral rules.
Consequentialism judges actions by their outcomes.
Virtue ethics focuses on the character of the moral agent.
Autonomy means respecting the right of individuals to make their own choices.
Beneficence means acting in the best interest of others.
Justice requires fair distribution of benefits and burdens.
Non-maleficence means do no harm.
Informed consent means people should know what they are agreeing to.
Transparency builds trust. Accountability means taking responsibility for actions.
""",
    "science": """
Biology: DNA contains genetic code. Neurons communicate via synapses. The brain has about 86 billion neurons.
Physics: E = mc^2. Energy equals mass times the speed of light squared. Gravity pulls objects together.
Chemistry: Water is H2O. Two hydrogen atoms bonded to one oxygen atom. Salt is NaCl.
Astronomy: The Earth orbits the Sun. The Sun is a star. The Milky Way is a galaxy.
Biology is the study of life. Cells are the basic unit of life.
DNA stores genetic information in a double helix structure.
RNA translates genetic code into proteins. Proteins perform most cellular functions.
Mitochondria generate energy through cellular respiration.
Natural selection drives evolution over generations.
Genes are segments of DNA that code for specific traits.
The nervous system transmits signals through neurons and synapses.
Photosynthesis converts sunlight into chemical energy in plants.
Ecosystems are communities of interacting organisms and their environment.
Physics describes the fundamental laws of the universe.
Newton's laws govern motion and forces. Thermodynamics studies heat and energy.
Electromagnetism unifies electricity and magnetism.
Quantum mechanics describes behavior at atomic and subatomic scales.
Special relativity shows that time and space are relative to the observer.
General relativity describes gravity as curvature of spacetime.
Chemistry studies the composition, structure, and properties of matter.
Atoms bond to form molecules through chemical bonds.
The periodic table organizes elements by atomic number and properties.
Acids have a pH below 7. Bases have a pH above 7.
""",
    "philosophy": """
Philosophy asks fundamental questions about existence, knowledge, and ethics.
Epistemology studies the nature of knowledge and belief.
Metaphysics studies the nature of reality and being.
Logic is the study of valid reasoning and argumentation.
A syllogism is a logical argument with two premises and a conclusion.
Deductive reasoning moves from general principles to specific conclusions.
Inductive reasoning moves from specific observations to general principles.
The Socratic method uses questioning to stimulate critical thinking.
Phenomenology studies the structure of conscious experience.
Existentialism emphasizes individual freedom and choice.
Stoicism teaches that virtue is the highest good and we should focus on what we control.
The mind-body problem asks how mental states relate to physical states.
Consciousness remains one of the deepest mysteries in philosophy.
""",
    "computing": """
Computer science is the study of computation and information processing.
A central processing unit executes instructions fetched from memory.
Memory stores data and instructions in binary form using bits and bytes.
An operating system manages hardware resources and provides services to programs.
A file system organizes data into files and directories on storage devices.
Networks connect computers to share data and resources.
The internet is a global network of interconnected computer networks.
TCP/IP is the fundamental protocol suite of the internet.
HTTP is the protocol used for transferring web pages.
Encryption secures data by transforming it into an unreadable form.
Public key cryptography uses a pair of keys for encryption and decryption.
Machine learning enables computers to learn from data without explicit programming.
Neural networks are computing systems inspired by biological neural networks.
Deep learning uses multiple layers of neural networks for hierarchical feature learning.
Reinforcement learning trains agents through rewards and punishments.
""",
    "neuroscience": """
The brain is the most complex organ in the human body.
Neurons are the basic building blocks of the nervous system.
A neuron receives signals through dendrites and sends signals through an axon.
Synapses are the junctions where neurons communicate with each other.
Neurotransmitters are chemical messengers that transmit signals across synapses.
Dopamine is involved in reward, motivation, and motor control.
Serotonin regulates mood, appetite, and sleep.
The cerebral cortex is responsible for higher cognitive functions.
The hippocampus is crucial for forming new memories.
The amygdala processes emotions, especially fear and pleasure.
Neuroplasticity is the brain's ability to reorganize itself by forming new neural connections.
Long-term potentiation strengthens synapses through repeated activation.
Working memory holds information temporarily for manipulation.
Sleep is essential for memory consolidation and learning.
Attention selects which information to process deeply and which to ignore.
Consciousness emerges from the integrated activity of billions of neurons.
The default mode network is active when the mind is at rest and wandering.
""",
}

all_seed_text = "\n".join(SEED_TEXTS.values())

print("\n[STAGE 2] Training BPE tokenizer...")

tokenizer_path = "bpe_vocab.json"
tokenizer = BPETokenizer(vocab_size=4096)

if os.path.exists(tokenizer_path):
    tokenizer.load(tokenizer_path)
    print(f"  Loaded existing BPE tokenizer: {tokenizer.get_vocab_size()} tokens")
else:
    tokenizer.train(all_seed_text)
    tokenizer.save(tokenizer_path)
    print(f"  BPE Vocabulary: {tokenizer.get_vocab_size()} tokens")

special_token_ids = tokenizer.SPECIAL_TOKENS
vocab_size = tokenizer.get_vocab_size()


class PlasticSynapse(nn.Module):
    """Synapse with meta-learned adaptive plasticity — rate changes based on task."""
    def __init__(self, in_features, out_features, plasticity_rate=0.01, use_meta=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.05)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.use_meta = use_meta
        if use_meta:
            self.meta_base = nn.Parameter(torch.tensor(plasticity_rate))
            self.meta_surprise_scale = nn.Parameter(torch.tensor(1.0))
            self.meta_gate = nn.Parameter(torch.tensor(0.0))
            self.meta_decay = nn.Parameter(torch.tensor(-3.0))
            self.meta_act_scale = nn.Parameter(torch.tensor(0.1))
        else:
            self.plasticity_rate = plasticity_rate
            self.weight_decay = 0.0005
        self.register_buffer('trace_pre', torch.zeros(in_features))
        self.register_buffer('trace_post', torch.zeros(out_features))
        self.register_buffer('trace_activation', torch.tensor(0.0), persistent=False)

    def get_plasticity_rate(self, surprise, input_activity=None, task_bias=None):
        """Compute plasticity rate from meta-params + surprise + activity + task bias."""
        if not self.use_meta:
            return self.plasticity_rate * surprise
        act = (input_activity or self.trace_activation)
        tb = task_bias if task_bias is not None else 0.0
        gate = torch.sigmoid(self.meta_gate + tb + self.meta_surprise_scale * (surprise - 0.5) + self.meta_act_scale * act)
        base = torch.nn.functional.softplus(self.meta_base)
        rate = base * gate
        return rate.clamp(1e-5, 0.2)

    def get_weight_decay(self):
        if not self.use_meta:
            return self.weight_decay
        return torch.sigmoid(self.meta_decay).item() * 0.01

    def hebbian_update(self, pre_activity, post_activity, surprise=1.0, input_activity=None, task_bias=None):
        if self.training:
            local_lr = self.get_plasticity_rate(surprise, input_activity, task_bias).detach()
            wd = self.get_weight_decay()
            pre = pre_activity.mean(dim=0).detach().flatten()
            post = post_activity.mean(dim=0).detach().flatten()
            pn, on = pre.norm(), post.norm()
            if pn > 0 and on > 0:
                pre = pre / (pn + 1e-8)
                post = post / (on + 1e-8)
            hebbian_delta = torch.outer(pre, post)
            if hebbian_delta.shape != self.weight.shape:
                hebbian_delta = hebbian_delta.view(self.weight.shape)
            oja_correction = hebbian_delta * (self.weight.detach() ** 2).mean().clamp(max=10)
            update = local_lr * (hebbian_delta - oja_correction)
            update = update.clamp(min=-1.0, max=1.0)
            self.weight.data.add_(update)
            self.weight.data.clamp_(min=-2.0, max=2.0)

    def forward(self, x):
        if self.training and x.dim() > 1:
            act = x.detach()
            self.trace_pre = act.mean(dim=(0, 1))
            self.trace_activation = act.abs().mean()
        out = x @ self.weight + self.bias
        if self.training and out.dim() > 1:
            self.trace_post = out.mean(dim=(0, 1)).detach()
        return out

    def get_meta_summary(self):
        if not self.use_meta:
            return {'plasticity_rate': self.plasticity_rate}
        with torch.no_grad():
            return {
                'base': self.meta_base.item(),
                'surprise_scale': self.meta_surprise_scale.item(),
                'gate': self.meta_gate.item(),
                'gate_gated': torch.sigmoid(self.meta_gate).item(),
                'decay': self.get_weight_decay(),
                'act_scale': self.meta_act_scale.item()
            }


class BiologicBlockV2(nn.Module):
    """Transformer block with sliding window attention + plastic feed-forward."""
    def __init__(self, embed_dim, num_heads, window_size, forward_expansion=4, dropout=0.1):
        super().__init__()
        self.attention = ChunkedSlidingAttention(embed_dim, num_heads, window_size, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            PlasticSynapse(embed_dim, embed_dim * forward_expansion),
            nn.ReLU(),
            PlasticSynapse(embed_dim * forward_expansion, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x


class BiologicLLMV2(nn.Module):
    """
    V2 model with BPE tokenization, 5000+ token sliding window context,
    plastic synapses, value system, GPU batch generation, and human-like gestalt.
    """

    def __init__(self, vocab_size, embed_dim=256, num_heads=8, num_layers=8,
                 max_context=5120, window_size=512, dropout=0.1, device=None):
        super().__init__()

        self.device = device or DEVICE
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_context = max_context
        self.window_size = window_size

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_context, embed_dim)

        self.blocks = nn.ModuleList([
            BiologicBlockV2(embed_dim, num_heads, window_size, 4, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

        self.eos_token_id = 3
        self.bos_token_id = 2

        self.value_network = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()
        )

        self.register_buffer('total_experience', torch.tensor(0))
        self.register_buffer('curiosity_level', torch.tensor(0.3))

        self.experience_buffer = deque(maxlen=2000)
        self.long_term_memory = []
        self.learning_rate_history = []
        self.surprise_history = []
        self.loss_history = []
        self.task_type_history = []

        self._optimizer = None
        self._nan_streak = 0

        self.task_profiles = {}

        # Task embeddings for task-conditioned plasticity
        all_tasks = sorted(SEED_TEXTS.keys()) + ['wiki', 'teach', 'ask', 'general']
        self.task_label_map = {name: i for i, name in enumerate(all_tasks)}
        self.task_embeddings = nn.Embedding(len(all_tasks), 16)
        self.task_projection = nn.Linear(16, 1)

        self._init_weights()
        self.to(self.device)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"  Neural architecture grown:")
        print(f"    - {num_layers} layers with sliding window attention (W={window_size})")
        print(f"    - Max context: {max_context} tokens")
        print(f"    - {num_heads} attention heads, {embed_dim} dimensions")
        print(f"    - {total_params:,} total parameters")
        print(f"    - Device: {self.device}")

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, return_value=False):
        B, T = idx.shape
        if T > self.max_context:
            idx = idx[:, -self.max_context:]
            T = self.max_context

        tok_emb = self.token_embedding(idx)
        pos = torch.arange(min(T, self.max_context), device=idx.device)
        pos_emb = self.position_embedding(pos).unsqueeze(0)
        x = self.dropout(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        if self.training:
            logits = torch.clamp(logits, min=-20.0, max=20.0)

        value = self.value_network(x.mean(dim=1)) if return_value else None

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            if targets.size(1) > T:
                targets = targets[:, :T]
            loss = F.cross_entropy(logits.reshape(-1, C), targets.reshape(-1))

        return logits, loss, value

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=200, temperature=0.5,
                 top_k=60, repetition_penalty=1.05):
        """Standard single-thread generation."""
        self.eval()
        device = self.device

        context = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generated = list(prompt_ids)

        for step in range(max_new_tokens):
            if context.size(1) > self.max_context:
                context = context[:, -self.max_context:]

            logits, _, value = self(context, return_value=True)
            logits = logits[:, -1, :] / temperature
            logits = torch.clamp(logits, min=-50.0, max=50.0)

            if value is not None:
                ethical_bias = value.item() * 0.3
                logits = logits + ethical_bias * 0.1

            if top_k > 0:
                k = min(top_k, logits.size(-1))
                values, indices = torch.topk(logits, k)
                threshold = values[:, -1:].expand_as(logits)
                logits = torch.where(logits < threshold, float('-inf'), logits)

            if repetition_penalty > 1.0:
                for prev_id in set(generated[-min(50, len(generated)):]):
                    if prev_id < logits.size(-1):
                        logits[0, prev_id] /= repetition_penalty

            if torch.all(logits == float('-inf')):
                logits = torch.zeros_like(logits) - 10.0

            logits = logits - logits.max(dim=-1, keepdim=True)[0]
            logits = torch.clamp(logits, min=-30.0, max=0.0)
            probs = F.softmax(logits, dim=-1)

            if torch.isnan(probs).any():
                probs = torch.ones_like(probs) / probs.size(-1)

            idx_next = torch.multinomial(probs, num_samples=1)
            next_idx = idx_next.item()
            generated.append(next_idx)
            context = torch.cat((context, idx_next), dim=1)

            if next_idx == self.eos_token_id:
                break

        return generated

    @torch.no_grad()
    def generate_human(self, prompt_ids, max_new_tokens=200,
                       gestalt_temp=1.0, main_temp=0.5, inner_temp=0.7,
                       top_k=60, repetition_penalty=1.05):
        """
        Human-like generation in three phases:
        1. Gestalt — high-temp direction finding (10 tokens)
        2. Main — fill in details at moderate temp
        3. Inner monologue — brief self-commentary at higher temp
        """
        self.eval()
        device = self.device
        full_context = list(prompt_ids)

        # Phase 1: Gestalt — high temperature direction-finding
        context_t = torch.tensor([full_context], dtype=torch.long, device=device)
        for _ in range(8):
            if context_t.size(1) > self.max_context:
                context_t = context_t[:, -self.max_context:]
            logits, _, _ = self(context_t)
            logits = logits[:, -1, :] / gestalt_temp
            logits = torch.clamp(logits, min=-50.0, max=50.0)
            probs = F.softmax(logits - logits.max(dim=-1, keepdim=True)[0], dim=-1)
            if torch.isnan(probs).any():
                probs = torch.ones_like(probs) / probs.size(-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            next_id = idx_next.item()
            full_context.append(next_id)
            context_t = torch.cat((context_t, idx_next), dim=1)
            if next_id == self.eos_token_id:
                break

        # Phase 2: Main generation
        for step in range(max_new_tokens - 20):
            if context_t.size(1) > self.max_context:
                context_t = context_t[:, -self.max_context:]
            logits, _, _ = self(context_t)
            logits = logits[:, -1, :] / main_temp
            logits = torch.clamp(logits, min=-50.0, max=50.0)

            if top_k > 0:
                k = min(top_k, logits.size(-1))
                vals, idxs = torch.topk(logits, k)
                logits = torch.where(logits < vals[:, -1:].expand_as(logits), float('-inf'), logits)

            if repetition_penalty > 1.0:
                for prev_id in set(full_context[-min(50, len(full_context)):]):
                    if prev_id < logits.size(-1):
                        logits[0, prev_id] /= repetition_penalty

            if torch.all(logits == float('-inf')):
                logits = torch.zeros_like(logits) - 10.0
            logits = logits - logits.max(dim=-1, keepdim=True)[0]
            logits = torch.clamp(logits, min=-30.0, max=0.0)
            probs = F.softmax(logits, dim=-1)
            if torch.isnan(probs).any():
                probs = torch.ones_like(probs) / probs.size(-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            next_id = idx_next.item()
            full_context.append(next_id)
            context_t = torch.cat((context_t, idx_next), dim=1)
            if next_id == self.eos_token_id:
                break

        # Phase 3: Inner monologue
        inner_prompt = full_context + tokenizer.encode(" (I realize that")[0:1]
        context_i = torch.tensor([inner_prompt[-self.max_context:]], dtype=torch.long, device=device)
        for _ in range(10):
            logits, _, _ = self(context_i)
            logits = logits[:, -1, :] / inner_temp
            logits = torch.clamp(logits, min=-50.0, max=50.0)
            probs = F.softmax(logits - logits.max(dim=-1, keepdim=True)[0], dim=-1)
            if torch.isnan(probs).any():
                probs = torch.ones_like(probs) / probs.size(-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            next_id = idx_next.item()
            full_context.append(next_id)
            context_i = torch.cat((context_i, idx_next), dim=1)
            if next_id == self.eos_token_id:
                break

        return full_context

    @torch.no_grad()
    def generate_batch(self, prompts, max_new_tokens=150, temperature=0.8,
                       top_k=50, repetition_penalty=1.05):
        """
        Batch generation for multiple prompts on GPU.
        Each prompt is encoded and generated independently.
        Returns list of decoded texts.
        """
        self.eval()
        if not isinstance(prompts, list):
            prompts = [prompts]

        results = []
        for prompt_text in prompts:
            prompt_ids = prompt_text if isinstance(prompt_text, list) else self._encode(prompt_text)
            gen_ids = self.generate(
                prompt_ids, max_new_tokens=max_new_tokens,
                temperature=temperature, top_k=top_k,
                repetition_penalty=repetition_penalty
            )
            results.append(gen_ids)

        return results

    def _encode(self, text):
        """Encode text to token IDs using module-level tokenizer."""
        return tokenizer.encode(text)

    def learn_from_interaction(self, input_ids, target_ids=None, value_label=None, task_type="general"):
        """Learn continuously from tokenized interactions.
        Task-aware: plasticity adapts per task type via meta-learned params.
        """
        self.train()
        self.total_experience += 1
        self.task_type_history.append(task_type)

        # Track task profile
        if task_type not in self.task_profiles:
            self.task_profiles[task_type] = {'losses': [], 'surprises': [], 'count': 0, 'best_loss': float('inf')}
        profile = self.task_profiles[task_type]

        if target_ids is not None:
            inp = torch.tensor([input_ids], dtype=torch.long, device=self.device)
            tgt = torch.tensor([target_ids], dtype=torch.long, device=self.device)

            logits, loss, value = self(inp, targets=tgt, return_value=True)

            if loss is not None and (torch.isnan(loss) or torch.isinf(loss)):
                # Clear NaN traces so they don't cascade into next step
                for module in self.modules():
                    if isinstance(module, PlasticSynapse):
                        module.trace_pre.zero_()
                        module.trace_post.zero_()
                        module.trace_activation.zero_()
                return {'loss': float('nan'), 'surprise': 5.0, 'value': 0}

            if loss is not None:
                surprise = min(5.0, max(0, (loss.item() - 0.5) * 2))
                self.surprise_history.append(surprise)
                self.loss_history.append(loss.item())
                profile['losses'].append(loss.item())
                profile['surprises'].append(surprise)
                profile['count'] += 1
                if loss.item() < profile['best_loss']:
                    profile['best_loss'] = loss.item()

                if self._optimizer is None:
                    self._create_optimizer()

                # Task embedding for task-conditioned plasticity
                task_id = self.task_label_map.get(task_type, self.task_label_map.get('general', 0))
                task_t = torch.tensor([task_id], dtype=torch.long, device=self.device)
                task_emb = self.task_embeddings(task_t)
                task_bias = self.task_projection(task_emb).squeeze()

                total_loss = loss
                if value_label is not None and value is not None:
                    value_target = torch.tensor([[value_label]], dtype=torch.float32, device=self.device)
                    value_loss = F.mse_loss(value, value_target)
                    total_loss = loss + 0.1 * value_loss

                # Gradient-based meta-learning: plasticity regularization gives meta-params gradients
                plasticity_penalty = self.plasticity_regularization(surprise)
                total_loss = total_loss + 0.001 * plasticity_penalty

                # Task-aware learning rate: adjust LR based on task profile
                task_lr_scale = 1.0
                if profile['count'] >= 5:
                    recent = np.mean(profile['losses'][-5:])
                    if recent > profile['best_loss'] * 1.2:
                        task_lr_scale = 1.5
                    elif recent < profile['best_loss'] * 0.9:
                        task_lr_scale = 0.8
                for pg in self._optimizer.param_groups:
                    pg['lr'] = pg.get('_base_lr', 5e-5) * task_lr_scale

                self._optimizer.zero_grad()
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                if torch.isnan(grad_norm) or grad_norm > 10 or grad_norm < 1e-8:
                    self._optimizer.zero_grad()
                    # Reset meta-params to safe defaults if they corrupted
                    self._nan_streak += 1
                    if self._nan_streak >= 3:
                        for module in self.modules():
                            if isinstance(module, PlasticSynapse) and module.use_meta:
                                module.meta_base.data.fill_(0.01)
                                module.meta_surprise_scale.data.fill_(1.0)
                                module.meta_gate.data.zero_()
                                module.meta_decay.data.fill_(-3.0)
                                module.meta_act_scale.data.fill_(0.1)
                        self._nan_streak = 0
                    self.surprise_history.append(5.0)
                    self.loss_history.append(loss.item() if loss is not None else 10)
                    # Still store in buffer even on bad grad (memory of the attempt)
                    self.experience_buffer.append({
                        'loss': loss.item() if loss is not None else 10,
                        'surprise': 5.0,
                        'value': 0,
                        'task': task_type,
                        'time': time.time(),
                        'input_ids': input_ids[:32],
                    })
                    return {'loss': loss.item() if loss is not None else 10, 'surprise': 5.0, 'value': 0}
                self._nan_streak = 0
                self._optimizer.step()

                # Hebbian updates with adaptive plasticity per synapse
                for block in self.blocks:
                    for module in block.modules():
                        if isinstance(module, PlasticSynapse) and hasattr(module, 'trace_pre'):
                            pre = module.trace_pre
                            post = module.trace_post
                            if pre is not None and post is not None and pre.dim() == 1 and post.dim() == 1:
                                if pre.norm() > 0 and post.norm() > 0:
                                    module.hebbian_update(
                                        pre.unsqueeze(0), post.unsqueeze(0),
                                        surprise=surprise, input_activity=module.trace_activation,
                                        task_bias=task_bias
                                    )

                self.experience_buffer.append({
                    'loss': loss.item(),
                    'surprise': surprise,
                    'value': value.item() if value is not None else 0,
                    'task': task_type,
                    'time': time.time(),
                    'input_ids': input_ids[:32],  # store for sleep replay
                })

                return {'loss': loss.item(), 'surprise': surprise, 'value': value.item() if value is not None else 0}

        return {'loss': None, 'surprise': 0, 'value': 0}

    def _create_optimizer(self):
        """Create AdamW with separate high-LR group for differentiable meta-params."""
        base_lr = 5e-5
        meta_keywords = ['meta_base', 'meta_gate', 'meta_surprise_scale', 'meta_decay', 'meta_act_scale', 'task_embedding', 'task_projection']
        meta_params = []
        model_params = []
        for n, p in self.named_parameters():
            is_meta = any(kw in n for kw in meta_keywords)
            is_excluded = 'value_network' in n or 'lm_head' in n
            if is_meta:
                meta_params.append(p)
            elif not is_excluded:
                model_params.append(p)
        self._optimizer = torch.optim.AdamW([
            {'params': model_params, 'lr': base_lr, '_base_lr': base_lr},
            {'params': self.lm_head.parameters(), 'lr': 1e-4, '_base_lr': 1e-4},
            {'params': self.value_network.parameters(), 'lr': 3e-4, '_base_lr': 3e-4},
            {'params': meta_params, 'lr': 5e-4, '_base_lr': 5e-4, 'weight_decay': 0},
        ], weight_decay=0.01)

    def plasticity_regularization(self, surprise):
        """Compute differentiable plasticity penalty for gradient-based meta-learning.
        Returns mean plasticity rate across all PlasticSynapses.
        Meta-params that produce higher plasticity get penalized more,
        encouraging them to find efficient plasticity levels through gradient descent.
        """
        reg = torch.tensor(0.0, device=self.device)
        count = 0
        for module in self.modules():
            if isinstance(module, PlasticSynapse) and module.use_meta:
                rate = module.get_plasticity_rate(surprise, input_activity=None)
                if torch.isfinite(rate).all():
                    reg = reg + rate.mean()
                    count += 1
        if count > 0 and torch.isfinite(reg).all():
            return reg / count
        return torch.tensor(0.0, device=self.device)

    def consolidate_memory(self):
        """Sleep cycle: synaptic rewiring (prune, break, join) + memory replay.
        Mimics human sleep: pruning weak connections, breaking overused ones,
        joining redundant paths, and replaying memories.
        """
        if len(self.experience_buffer) < 10:
            return

        print(f"\n  [SLEEP] Synaptic rewiring {len(self.experience_buffer)} memories...")
        stats = {'pruned': 0, 'broken': 0, 'joined': 0, 'replayed': 0}

        # --- Phase 1: Prune weak synapses ---
        for module in self.modules():
            if isinstance(module, PlasticSynapse):
                with torch.no_grad():
                    w = module.weight.data
                    threshold = w.abs().mean() * 0.15
                    weak = w.abs() < threshold
                    stats['pruned'] += weak.sum().item()
                    w[weak] = 0

        # --- Phase 2: Break overactive synapses (add noise to diversify) ---
        for module in self.modules():
            if isinstance(module, PlasticSynapse):
                with torch.no_grad():
                    w = module.weight.data
                    mu, sigma = w.abs().mean(), w.abs().std()
                    active = w.abs() > (mu + sigma)
                    noise = torch.randn_like(w) * 0.01 * self.curiosity_level
                    w[active] += noise[active]
                    stats['broken'] += active.sum().item()

        # --- Phase 3: Join redundant weight vectors ---
        for module in self.modules():
            if isinstance(module, PlasticSynapse):
                with torch.no_grad():
                    w = module.weight.data
                    if w.dim() == 2 and w.size(0) > 1 and w.size(1) > 1:
                        w_norm = w / (w.norm(dim=1, keepdim=True) + 1e-8)
                        sim = w_norm @ w_norm.T
                        sim.fill_diagonal_(0)
                        pairs = (sim > 0.95).nonzero(as_tuple=False)
                        if pairs.size(0) > 0:
                            for p in pairs[:min(8, len(pairs))]:
                                avg = (w[p[0]] + w[p[1]]) / 2
                                noise = torch.randn_like(avg) * 0.005
                                w[p[0]] = avg + noise
                                w[p[1]] = avg - noise
                            stats['joined'] += pairs.size(0)

        # --- Phase 4: Replay high-surprise memories with Hebbian updates ---
        replay_sorted = sorted(
            self.experience_buffer,
            key=lambda x: abs(x.get('surprise', 0)),
            reverse=True
        )[:16]

        self.train()
        for mem in replay_sorted:
            inp_ids = mem.get('input_ids')
            if inp_ids and len(inp_ids) >= 4:
                try:
                    inp = torch.tensor([inp_ids[:16]], dtype=torch.long, device=self.device)
                    with torch.no_grad():
                        _ = self(inp, targets=None)
                    # Hebbian update from the forward traces
                    surprise = abs(mem.get('surprise', 1.0))
                    for block in self.blocks:
                        for module in block.modules():
                            if isinstance(module, PlasticSynapse):
                                pre = module.trace_pre
                                post = module.trace_post
                                if pre is not None and post is not None and pre.dim() == 1 and post.dim() == 1:
                                    if pre.norm() > 0 and post.norm() > 0:
                                        module.hebbian_update(
                                            pre.unsqueeze(0), post.unsqueeze(0),
                                            surprise=surprise, input_activity=module.trace_activation
                                        )
                    stats['replayed'] += 1
                except:
                    pass

        # --- Phase 5: Consolidate buffer (keep highest-surprise items) ---
        self.experience_buffer = deque(
            sorted(self.experience_buffer,
                   key=lambda x: abs(x.get('surprise', 0)), reverse=True)[:1000],
            maxlen=2000
        )

        self.curiosity_level = min(0.8, self.curiosity_level + 0.02)

        total_synapses = sum(
            m.weight.numel() for m in self.modules() if isinstance(m, PlasticSynapse)
        )
        sparsity = stats['pruned'] / max(1, total_synapses) * 100
        print(
            f"  [SLEEP] Pruned {stats['pruned']} synapses ({sparsity:.1f}%) | "
            f"Broke {stats['broken']} | Joined {stats['joined']} pairs | "
            f"Replayed {stats['replayed']} memories"
        )
        print(f"  [SLEEP] Buffer: {len(self.experience_buffer)} | Curiosity: {self.curiosity_level:.2f}")

    def self_improve(self):
        """Meta-controller: analyze and suggest improvements."""
        if len(self.surprise_history) < 10:
            return None

        recent_surprise = np.mean(self.surprise_history[-50:]) if len(self.surprise_history) >= 50 else np.mean(self.surprise_history)
        avg_loss = np.nanmean(self.loss_history[-50:]) if len(self.loss_history) >= 50 else (1.0 if not self.loss_history else np.nanmean(self.loss_history))

        improvements = []
        if avg_loss > 2.0:
            improvements.append("Increase model capacity")
        if recent_surprise < 0.1:
            improvements.append("Raise curiosity - too predictable")
            self.curiosity_level = min(1.0, self.curiosity_level + 0.1)
        if recent_surprise > 5.0:
            improvements.append("Reduce plasticity - too volatile")

        assessment = {
            'timestamp': datetime.now().isoformat(),
            'avg_surprise': float(recent_surprise),
            'avg_loss': float(avg_loss),
            'curiosity': float(self.curiosity_level),
            'buffer_size': len(self.experience_buffer),
            'total_experience': int(self.total_experience),
            'suggestions': improvements
        }

        try:
            assessments = []
            if os.path.exists('self_assessment.json'):
                with open('self_assessment.json', 'r') as f:
                    assessments = json.load(f)
            assessments.append(assessment)
            with open('self_assessment.json', 'w') as f:
                json.dump(assessments, f, indent=2)
        except:
            pass

        return assessment

    def get_state_summary(self):
        meta_info = {}
        meta_synapses = [m for m in self.modules() if isinstance(m, PlasticSynapse) and m.use_meta]
        if meta_synapses:
            meta_info['meta_base_range'] = f"{min(s.meta_base.item() for s in meta_synapses):.4f}-{max(s.meta_base.item() for s in meta_synapses):.4f}"
            avg_gate = np.mean([torch.sigmoid(s.meta_gate).item() for s in meta_synapses])
            meta_info['avg_gate'] = f"{avg_gate:.3f}"
            meta_info['surprise_scale_range'] = f"{min(s.meta_surprise_scale.item() for s in meta_synapses):.2f}-{max(s.meta_surprise_scale.item() for s in meta_synapses):.2f}"
        return {
            'total_experience': int(self.total_experience),
            'curiosity_level': float(self.curiosity_level),
            'buffer_size': len(self.experience_buffer),
            'recent_surprise': float(np.mean(self.surprise_history[-20:])) if self.surprise_history else 0,
            'avg_loss': float(np.nanmean(self.loss_history[-20:])) if self.loss_history else 0,
            'vocab_size': self.vocab_size,
            'max_context': self.max_context,
            'window_size': self.window_size,
            'device': str(self.device),
            'parameters': sum(p.numel() for p in self.parameters()),
            'plastics': sum(1 for m in self.modules() if isinstance(m, PlasticSynapse)),
            **meta_info
        }

    def get_task_summary(self):
        """Return per-task performance summary."""
        lines = []
        for task, prof in sorted(self.task_profiles.items()):
            if prof['count'] > 0:
                avg_l = np.nanmean(prof['losses']) if prof['losses'] else 0
                avg_s = np.nanmean(prof['surprises']) if prof['surprises'] else 0
                lines.append(f"  {task}: {prof['count']} steps, loss={avg_l:.2f}, surprise={avg_s:.2f}, best={prof['best_loss']:.2f}")
        return '\n'.join(lines)


def create_model(vocab_size=None, embed_dim=None, num_heads=8, num_layers=None,
                  max_context=5120, window_size=512, dropout=0.1, do_seed_learning=True,
                  tokenizer_ref=None, auto_scale=True):
    """Factory function: auto-scale for GPU, create and seed-train."""
    if vocab_size is None:
        global tokenizer
        vocab_size = tokenizer.get_vocab_size()

    is_gpu = torch.cuda.is_available()

    if auto_scale and is_gpu:
        embed_dim = embed_dim or 512
        num_layers = num_layers or 12
        print(f"\n[GPU] Auto-scaling: {embed_dim}d, {num_layers} layers")
    else:
        embed_dim = embed_dim or 256
        num_layers = num_layers or 8
        print(f"\n  Using: {embed_dim}d, {num_layers} layers")

    print(f"\n[STAGE 3] V2 model coming online...")
    m = BiologicLLMV2(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_context=max_context,
        window_size=window_size,
        dropout=dropout,
        device=DEVICE
    )

    if tokenizer_ref is not None:
        m.eos_token_id = tokenizer_ref.SPECIAL_TOKENS.get('<EOS>', 3)
        m.bos_token_id = tokenizer_ref.SPECIAL_TOKENS.get('<BOS>', 2)

    print(f"\n  Total parameters: {sum(p.numel() for p in m.parameters()):,}")
    print(f"  Experience buffer: {m.experience_buffer.maxlen}")
    print(f"  Device: {m.device}")

    if do_seed_learning:
        print(f"\n[STAGE 4] Seed learning phase...")
        for epoch in range(5):
            total_steps = 0
            for domain, text in SEED_TEXTS.items():
                encoded = tokenizer.encode(text)
                if len(encoded) < 4:
                    continue
                chunk_size = min(32, len(encoded) - 2)
                stride = max(4, chunk_size // 4)
                count = 0
                for i in range(0, len(encoded) - chunk_size - 1, stride):
                    chunk = encoded[i:i + chunk_size]
                    target = encoded[i + 1:i + chunk_size + 1]
                    if len(chunk) == len(target) and len(chunk) > 1:
                        m.learn_from_interaction(chunk, target, value_label=0.3, task_type=domain)
                        count += 1
                total_steps += count
                if epoch == 0:
                    print(f"  {domain}: {count} steps")
            print(f"  Epoch {epoch+1}/5: {total_steps} steps, exp={m.total_experience}")
        print(f"[STAGE 4] Seed learning complete. Experiences: {m.total_experience}")

        # Wikipedia seed knowledge
        wiki_count = 0
        try:
            from web_learner import WebLearner
            wl = WebLearner()
            wiki_topics = ["Neuron", "DNA", "Evolution", "Gravity", "Algorithm"]
            for topic in wiki_topics:
                try:
                    result = wl.learn(topic, max_chars=600)
                    if result['success']:
                        text = result['content']
                        encoded = tokenizer.encode(text)
                        if len(encoded) > 10:
                            chunk_size = min(32, len(encoded) - 2)
                            stride = max(4, chunk_size // 4)
                            for i in range(0, len(encoded) - chunk_size - 1, stride):
                                chunk = encoded[i:i + chunk_size]
                                target = encoded[i + 1:i + chunk_size + 1]
                                if len(chunk) == len(target) and len(chunk) > 1:
                                    m.learn_from_interaction(chunk, target, value_label=0.5, task_type=f"wiki:{topic}")
                                    wiki_count += 1
                except:
                    pass
            if wiki_count > 0:
                print(f"  Wikipedia seed: {wiki_count} additional steps")
        except ImportError:
            pass
        except Exception as e:
            print(f"  [WIKI] Skipped: {e}")

        print(f"\n[STAGE 6] First sleep cycle...")
        m.consolidate_memory()

    return m


if __name__ == "__main__":
    model = create_model(vocab_size=vocab_size, do_seed_learning=True)

    print("\n" + "=" * 60)
    print("BIOLOGIC LLM V2 IS NOW ONLINE")
    print(f"Device: {DEVICE} | Context: {model.max_context} | BPE: {vocab_size}")
    print("=" * 60)
    print()
