"""
DREAM ENGINE — Dream generation during sleep cycles.
Three dream types:
  REMIX:      Blend 2 memories from different domains into novel scenarios
  COMPRESSION: Replay high-surprise memories to strengthen their anchors
  NOVELTY:    Explore lonely concepts and create new associations

During sleep, dreams produce synthetic training data the model learns from
and create creative cross-domain links in the memory graph.
"""

import time
import random
import json
import re
from datetime import datetime


class DreamEngine:
    """
    Generates and processes dreams during sleep cycles.
    Dreams recombine episodic memories, strengthen anchors, and forge new links.
    """

    def __init__(self, episodic_memory=None, memory_graph=None,
                 filepath="dream_log.json"):
        self.episodic = episodic_memory
        self.memory_graph = memory_graph
        self.filepath = filepath
        self.dream_history = []
        self.dream_count = 0

    def dream(self, model, tokenizer, dream_type="remix", temperature=1.0):
        """Generate a single dream of the specified type."""
        if dream_type == "remix":
            return self._dream_remix(model, tokenizer, temperature)
        elif dream_type == "compression":
            return self._dream_compression(model, tokenizer, temperature)
        elif dream_type == "novelty":
            return self._dream_novelty(model, tokenizer, temperature)
        else:
            return self._dream_remix(model, tokenizer, temperature)

    def dream_cycle(self, model, tokenizer, n_dreams=3):
        """Run a full dream cycle: 1 of each type, batched on GPU."""
        if not self.episodic or len(self.episodic.traces) < 3:
            return {"dreams": [], "new_links": [], "note": "Not enough memories to dream"}

        dream_types = ["remix", "compression", "novelty"]
        random.shuffle(dream_types)
        dream_types = dream_types[:n_dreams]

        results = []
        all_new_links = []

        for dtype in dream_types:
            result = self.dream(model, tokenizer, dtype)
            if result:
                results.append(result)
                all_new_links.extend(result.get('new_links', []))

        self.dream_count += len(results)

        log_entry = {
            'timestamp': time.time(),
            'dream_count': len(results),
            'types': [r.get('type') for r in results],
            'new_links': all_new_links,
        }
        self.dream_history.append(log_entry)
        self._save_log()

        return {
            'dreams': results,
            'new_links': all_new_links,
            'total_dreams': self.dream_count,
        }

    def _dream_remix(self, model, tokenizer, temperature=1.0):
        """Blend 2 memories from different domains into a novel dream."""
        traces = list(self.episodic.traces.values())
        if len(traces) < 2:
            return None

        a = random.choice(traces)
        b = random.choice([t for t in traces if t.trace_id != a.trace_id])

        a_concepts = a.anchor_concepts[:2] if a.anchor_concepts else ["something"]
        b_concepts = b.anchor_concepts[:2] if b.anchor_concepts else ["something else"]

        seed = (
            f"I had a dream where {a_concepts[0]} and {b_concepts[0]} "
            f"were connected in a strange way. "
            f"{a.context_snippet[:100]} "
            f"...and at the same time, {b.context_snippet[:100]} "
            f"In my dream, they merged into something new:"
        )

        prompt_ids = tokenizer.encode(seed)
        generated_ids = model.generate(
            prompt_ids, max_new_tokens=200, temperature=temperature,
            top_k=40, repetition_penalty=1.0
        )
        dream_text = tokenizer.decode(generated_ids)

        new_links = []
        if self.memory_graph and a_concepts and b_concepts:
            link = self.memory_graph.add_link(
                a_concepts[0], b_concepts[0],
                link_type="dreamt_together",
                weight=0.5
            )
            new_links.append(f"{a_concepts[0]} --[dreamt_together]--> {b_concepts[0]}")

        self.episodic.store(
            dream_text,
            concepts=list(set(a_concepts + b_concepts + ["dream", "connection"])),
            valence=0.3,
            source="dream_remix"
        )

        return {
            'type': 'remix',
            'source_a': a_concepts,
            'source_b': b_concepts,
            'dream_text': dream_text,
            'new_links': new_links,
        }

    def _dream_compression(self, model, tokenizer, temperature=0.9):
        """Replay and compress high-surprise memories."""
        traces = self.episodic.highest_surprise(n=3)
        if not traces:
            return None

        target = traces[0]
        target.recall()

        seed = (
            f"I keep thinking about {', '.join(target.anchor_concepts[:3])}. "
            f"{target.context_snippet[:120]} "
            f"The key insight I keep coming back to is:"
        )

        prompt_ids = tokenizer.encode(seed)
        generated_ids = model.generate(
            prompt_ids, max_new_tokens=150, temperature=temperature,
            top_k=50,
        )
        dream_text = tokenizer.decode(generated_ids)

        target.activation = min(10.0, target.activation * 1.5)

        return {
            'type': 'compression',
            'source': target.anchor_concepts,
            'dream_text': dream_text,
            'new_links': [],
        }

    def _dream_novelty(self, model, tokenizer, temperature=1.0):
        """Explore lonely concepts and create new associations."""
        traces = self.episodic.least_connected(n=3)
        if not traces:
            return None

        target = traces[0]
        concepts = target.anchor_concepts[:2] if target.anchor_concepts else ["unknown"]

        seed = (
            f"I wonder what {concepts[0]} could be connected to. "
            f"{target.context_snippet[:100]} "
            f"Maybe it relates to:"
        )

        prompt_ids = tokenizer.encode(seed)
        generated_ids = model.generate(
            prompt_ids, max_new_tokens=180, temperature=temperature,
            top_k=40, repetition_penalty=1.0
        )
        dream_text = tokenizer.decode(generated_ids)

        new_links = []
        if self.memory_graph and concepts:
            extracted = re.findall(r'[A-Z][a-z]+(?:\s[A-Z][a-z]+)?', dream_text)
            for term in extracted[:3]:
                term_lower = term.lower().strip()
                if (term_lower not in [c.lower() for c in concepts] and
                        len(term) > 3):
                    link = self.memory_graph.add_link(
                        concepts[0], term,
                        link_type="dreamt_exploration",
                        weight=0.3
                    )
                    new_links.append(f"{concepts[0]} --[dreamt_exploration]--> {term}")

        self.episodic.store(
            dream_text,
            concepts=concepts + ["dream", "exploration"],
            valence=0.2,
            source="dream_novelty"
        )

        return {
            'type': 'novelty',
            'source': concepts,
            'dream_text': dream_text,
            'new_links': new_links,
        }

    def learn_from_dreams(self, model, tokenizer, dream_results):
        """Have the model learn from its own dreams."""
        learned_count = 0
        for dream in dream_results:
            if not dream:
                continue
            text = dream.get('dream_text', '')
            if len(text) < 20:
                continue

            encoded = tokenizer.encode(text)
            chunk_size = min(32, len(encoded) - 1)
            if chunk_size < 4:
                continue

            for i in range(0, len(encoded) - chunk_size - 1, chunk_size // 2):
                chunk = encoded[i:i + chunk_size]
                target = encoded[i + 1:i + chunk_size + 1]
                if len(chunk) == len(target) and len(chunk) > 1:
                    model.learn_from_interaction(chunk, target, value_label=0.3)
                    learned_count += 1

        return learned_count

    def stats(self):
        if not self.dream_history:
            return "No dreams yet."
        return (
            f"Total dreams: {self.dream_count}\n"
            f"Dream sessions: {len(self.dream_history)}\n"
            f"Last dream: {datetime.fromtimestamp(self.dream_history[-1]['timestamp']).strftime('%H:%M') if self.dream_history else 'never'}"
        )

    def _save_log(self):
        try:
            with open(self.filepath, 'w') as f:
                json.dump(self.dream_history[-50:], f, indent=2)
        except:
            pass
