"""
EPISODIC MEMORY — Memory traces anchored to concepts with pattern completion.
Stores compressed experiences and reconstructs them fresh each time,
like how humans recall memories from a few sensory cues.
"""

import time
import json
import re
from collections import defaultdict
from datetime import datetime


class MemoryTrace:
    """A single episodic memory stored as an anchor + compressed trace."""

    _id_counter = 0

    def __init__(self, anchor_concepts, context_snippet, valence=0.0,
                 source="user", timestamp=None, trace_id=None):
        MemoryTrace._id_counter += 1
        self.trace_id = trace_id or f"{time.time()}_{MemoryTrace._id_counter}"
        self.anchor_concepts = list(set(c.lower().strip() for c in anchor_concepts if c))
        self.context_snippet = context_snippet[:200]
        self.valence = max(-1.0, min(1.0, valence))
        self.source = source
        self.timestamp = timestamp or time.time()
        self.activation = 1.0
        self.reconstruction_count = 0
        self.last_recalled = 0

    def recall(self):
        self.reconstruction_count += 1
        self.activation = min(10.0, self.activation + 0.5)
        self.last_recalled = time.time()

    def decay(self, rate=0.01):
        self.activation = max(0.1, self.activation * (1 - rate))

    def strength(self):
        recency = max(0, 1.0 - (time.time() - self.last_recalled) / 86400)
        return self.activation * (0.3 + 0.7 * recency) * (1 + 0.2 * self.reconstruction_count)

    def to_dict(self):
        return {
            'trace_id': self.trace_id,
            'anchor_concepts': self.anchor_concepts,
            'context_snippet': self.context_snippet,
            'valence': self.valence,
            'source': self.source,
            'timestamp': self.timestamp,
            'activation': self.activation,
            'reconstruction_count': self.reconstruction_count,
            'last_recalled': self.last_recalled
        }

    @staticmethod
    def from_dict(d):
        t = MemoryTrace(
            anchor_concepts=d.get('anchor_concepts', []),
            context_snippet=d.get('context_snippet', ''),
            valence=d.get('valence', 0.0),
            source=d.get('source', 'unknown'),
            timestamp=d.get('timestamp', time.time()),
            trace_id=d.get('trace_id')
        )
        t.activation = d.get('activation', 1.0)
        t.reconstruction_count = d.get('reconstruction_count', 0)
        t.last_recalled = d.get('last_recalled', 0)
        return t


class EpisodicMemory:
    """
    Manages episodic memory traces with pattern completion.
    Stores compressed anchors, reconstructs full experiences on recall.
    """

    def __init__(self, filepath="episodic_memory.json", max_traces=500):
        self.filepath = filepath
        self.max_traces = max_traces
        self.traces = {}
        self._concept_index = defaultdict(set)
        self.load()

    def store(self, text, concepts=None, valence=0.0, source="user"):
        if not text or len(text) < 10:
            return None

        anchor = concepts or self._extract_anchor_concepts(text)
        if not anchor:
            return None

        snippet = text[:200].strip()

        trace = MemoryTrace(
            anchor_concepts=anchor,
            context_snippet=snippet,
            valence=valence,
            source=source
        )

        self.traces[trace.trace_id] = trace
        for concept in anchor:
            self._concept_index[concept.lower()].add(trace.trace_id)

        if len(self.traces) > self.max_traces:
            self._prune()

        return trace

    def reconstruct(self, cue, model, tokenizer, temperature=0.85, max_new=200):
        traces = self.recall_similar(cue, top_k=3)
        if not traces:
            return None

        best = traces[0]
        best.recall()

        context = f"I remember learning about {', '.join(best.anchor_concepts[:3])}. "
        if best.context_snippet:
            context += f"It makes me think of {best.context_snippet[:100]}. "
        context += f"This feels {self._valence_desc(best.valence)}. "

        seed = f"Recalling from memory: {context}The full memory comes back to me now:"

        prompt_ids = tokenizer.encode(seed)
        generated_ids = model.generate(prompt_ids, max_new_tokens=max_new, temperature=temperature)
        reconstruction = tokenizer.decode(generated_ids)

        extra = ""
        if len(traces) > 1:
            extra = f"\n...this also connects to {traces[1].anchor_concepts[0] if traces[1].anchor_concepts else 'something else familiar'}"

        return {
            'anchor': best.anchor_concepts,
            'valence': best.valence,
            'reconstruction': reconstruction,
            'associated': extra,
            'strength': best.strength()
        }

    def pattern_complete(self, partial_cues, model, tokenizer, temperature=0.9, max_new=150):
        matches = self.recall_similar(partial_cues, top_k=5)
        if not matches:
            return None

        cue_list = [c.lower().strip() for c in re.split(r'[,\s]+', partial_cues) if len(c.strip()) > 2]

        scored = []
        for trace in matches:
            overlap = len(set(cue_list) & set(trace.anchor_concepts))
            scored.append((overlap, trace.strength(), trace))

        scored.sort(key=lambda x: (-x[0], -x[1]))
        best = scored[0][2]
        best.recall()

        seed = f"Given {' and '.join(cue_list[:3])}, I recall: {best.context_snippet[:100]}"

        prompt_ids = tokenizer.encode(seed)
        generated_ids = model.generate(prompt_ids, max_new_tokens=max_new, temperature=temperature)
        completed = tokenizer.decode(generated_ids)

        return {
            'cues': partial_cues,
            'matched_anchor': best.anchor_concepts,
            'completed': completed,
            'confidence': scored[0][0] / max(len(cue_list), 1)
        }

    def recall_similar(self, cue, top_k=5):
        if not self.traces:
            return []

        cue_lower = cue.lower().strip()
        cue_words = set(re.split(r'[^a-zA-Z0-9]+', cue_lower))
        cue_words = {w for w in cue_words if len(w) > 2}

        scored = []
        for trace in self.traces.values():
            concept_match = sum(1 for c in trace.anchor_concepts if c in cue_lower or any(cw in c for cw in cue_words))
            word_overlap = len(set(w for w in cue_words if any(w in ac for ac in trace.anchor_concepts)))
            score = concept_match * 2 + word_overlap + trace.strength() * 0.5
            if score > 0:
                scored.append((score, trace))

        scored.sort(key=lambda x: -x[0])
        return [t for _, t in scored[:top_k]]

    def recent_memories(self, n=10):
        sorted_traces = sorted(self.traces.values(), key=lambda t: -t.timestamp)
        return sorted_traces[:n]

    def highest_surprise(self, n=5):
        sorted_traces = sorted(
            self.traces.values(),
            key=lambda t: (abs(t.valence) * t.reconstruction_count if t.reconstruction_count > 0 else 0.1),
            reverse=True
        )
        return sorted_traces[:n]

    def least_connected(self, n=3):
        if not self.traces:
            return []
        sorted_traces = sorted(self.traces.values(), key=lambda t: len(t.anchor_concepts))
        return sorted_traces[:n]

    def stats(self):
        if not self.traces:
            return "Episodic memory is empty."

        avg_valence = sum(t.valence for t in self.traces.values()) / len(self.traces)
        avg_activation = sum(t.activation for t in self.traces.values()) / len(self.traces)
        total_recalled = sum(t.reconstruction_count for t in self.traces.values())

        return (
            f"Traces: {len(self.traces)}\n"
            f"Avg valence: {avg_valence:.2f}\n"
            f"Avg activation: {avg_activation:.2f}\n"
            f"Total reconstructions: {total_recalled}\n"
            f"Most recalled: {max(self.traces.values(), key=lambda t: t.reconstruction_count).anchor_concepts[:3]}"
        )

    def save(self):
        data = {
            'traces': {tid: t.to_dict() for tid, t in self.traces.items()},
            'updated': time.time()
        }
        try:
            with open(self.filepath, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"  [EPISODIC] Save error: {e}")
            return False

    def load(self):
        if not os.path.exists(self.filepath):
            return False
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
            self.traces = {}
            self._concept_index.clear()
            for tid, td in data.get('traces', {}).items():
                trace = MemoryTrace.from_dict(td)
                self.traces[tid] = trace
                for c in trace.anchor_concepts:
                    self._concept_index[c.lower()].add(tid)
            return True
        except Exception as e:
            print(f"  [EPISODIC] Load error: {e}")
            return False

    def _prune(self):
        sorted_traces = sorted(
            self.traces.values(),
            key=lambda t: t.strength()
        )
        to_remove = sorted_traces[:len(self.traces) - self.max_traces // 2]
        for trace in to_remove:
            self.traces.pop(trace.trace_id, None)
            for c in trace.anchor_concepts:
                self._concept_index[c.lower()].discard(trace.trace_id)

    def _extract_anchor_concepts(self, text):
        wiki = re.findall(r'\[\[(.*?)\]\]', text)
        if wiki:
            return wiki[:5]
        words = re.findall(r'[A-Z][a-z]+(?:\s[A-Z][a-z]+)?', text)
        words = [w.lower() for w in words if len(w) > 3]
        freq = defaultdict(int)
        for w in words:
            freq[w] += 1
        sorted_words = sorted(freq.items(), key=lambda x: -x[1])
        return [w for w, c in sorted_words[:5]]

    def _valence_desc(self, v):
        if v > 0.3: return "interesting and positive"
        if v < -0.3: return "concerning"
        return "neutral"


import os
