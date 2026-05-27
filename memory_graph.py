"""
MEMORY GRAPH — Obsidian-like bi-directional knowledge graph.
Persistent concept linking with [[wiki-link]] support, graph traversal, and search.
"""

import re
import json
import os
import time
from datetime import datetime
from collections import deque, defaultdict


class ConceptNode:
    """A single concept in the knowledge graph."""

    def __init__(self, name, aliases=None, notes="", first_seen=None):
        self.name = name
        self.aliases = aliases or []
        self.notes = notes
        self.first_seen = first_seen or time.time()
        self.last_seen = time.time()
        self.reference_count = 1

    def to_dict(self):
        return {
            'name': self.name,
            'aliases': self.aliases,
            'notes': self.notes,
            'first_seen': self.first_seen,
            'last_seen': self.last_seen,
            'reference_count': self.reference_count
        }

    @classmethod
    def from_dict(cls, d):
        node = cls(d['name'], d.get('aliases', []), d.get('notes', ''), d.get('first_seen'))
        node.last_seen = d.get('last_seen', time.time())
        node.reference_count = d.get('reference_count', 1)
        return node

    def reference(self):
        self.reference_count += 1
        self.last_seen = time.time()

    def __repr__(self):
        return f"Concept({self.name}, refs={self.reference_count})"


class ConceptLink:
    """A bi-directional edge between two concepts."""

    def __init__(self, source, target, link_type="related", weight=1.0):
        self.source = source.lower().strip()
        self.target = target.lower().strip()
        self.link_type = link_type
        self.weight = weight
        self.created = time.time()
        self.last_updated = time.time()

    def to_dict(self):
        return {
            'source': self.source,
            'target': self.target,
            'link_type': self.link_type,
            'weight': self.weight,
            'created': self.created,
            'last_updated': self.last_updated
        }

    @classmethod
    def from_dict(cls, d):
        link = cls(d['source'], d['target'], d.get('link_type', 'related'), d.get('weight', 1.0))
        link.created = d.get('created', time.time())
        link.last_updated = d.get('last_updated', time.time())
        return link

    def __repr__(self):
        return f"Link({self.source} --[{self.link_type}]--> {self.target})"


class WikiLinkParser:
    """Parses [[wiki-links]] from text."""

    WIKI_LINK_RE = re.compile(r'\[\[([^\]]+?)\]\]')

    @staticmethod
    def extract(text):
        """Returns list of link targets found in text."""
        return WikiLinkParser.WIKI_LINK_RE.findall(text)

    @staticmethod
    def has_links(text):
        return bool(WikiLinkParser.WIKI_LINK_RE.search(text))

    @staticmethod
    def strip_links(text):
        """Replace [[Concept]] with 'Concept' for display."""
        return WikiLinkParser.WIKI_LINK_RE.sub(r'\1', text)


class ConceptExtractor:
    """Auto-extract concepts from plain text (no [[wiki-links]])."""

    # Words to skip (not concepts)
    STOP_WORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
        'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
        'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
        'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and',
        'or', 'if', 'while', 'that', 'this', 'it', 'its', 'about', 'up',
        'like', 'which', 'what', 'who', 'whom', 'whose'
    }

    # Patterns for important concepts
    CAPITALIZED_PATTERN = re.compile(r'\b([A-Z][a-z]+(?:[ -][A-Z][a-z]+)*)\b')
    QUOTED_PATTERN = re.compile(r'"([^"]{3,})"')
    ACRONYM_PATTERN = re.compile(r'\b([A-Z]{2,})\b')

    @classmethod
    def extract(cls, text, min_freq=1):
        """Extract candidate concepts from text. Returns dict of {name: frequency}."""
        concepts = defaultdict(int)

        # Find capitalized phrases (proper nouns, key terms)
        for match in cls.CAPITALIZED_PATTERN.finditer(text):
            word = match.group(1)
            if word.lower() not in cls.STOP_WORDS and len(word) > 2:
                concepts[word] += 1

        # Find quoted phrases
        for match in cls.QUOTED_PATTERN.finditer(text):
            phrase = match.group(1)
            if phrase.lower() not in cls.STOP_WORDS and len(phrase) > 3:
                concepts[phrase] += 1

        # Find acronyms
        for match in cls.ACRONYM_PATTERN.finditer(text):
            acro = match.group(1)
            if len(acro) >= 2 and acro not in {'AI', 'BI', 'ID', 'OK'}:
                concepts[acro] += 1

        # Find interesting lowercase words (length >= 6, not in stop words)
        for match in re.finditer(r'\b([a-z]{6,})\b', text.lower()):
            word = match.group(1)
            if word not in cls.STOP_WORDS:
                concepts[word.title()] += 1

        # Filter by frequency
        return {k: v for k, v in concepts.items() if v >= min_freq}


class MemoryGraph:
    """
    Obsidian-like bi-directional knowledge graph.
    Stores concepts as nodes and relationships as edges.
    Supports [[wiki-links]], backlinks, search, and graph traversal.
    """

    def __init__(self, filepath="memory_graph.json"):
        self.filepath = filepath
        self.nodes = {}        # name.lower() -> ConceptNode
        self.edges = []        # list of ConceptLink
        self._adjacency = defaultdict(set)  # name.lower() -> set of neighbor names
        self._backlinks_cache = defaultdict(set)
        self.load()

    # ---- Core Operations ----

    def add_concept(self, name, aliases=None, notes=""):
        """Add a concept or update existing one."""
        key = name.lower().strip()
        if key in self.nodes:
            self.nodes[key].reference()
            if notes and not self.nodes[key].notes:
                self.nodes[key].notes = notes
            if aliases:
                for a in aliases:
                    if a.lower() not in [x.lower() for x in self.nodes[key].aliases]:
                        self.nodes[key].aliases.append(a)
        else:
            self.nodes[key] = ConceptNode(name.strip(), aliases, notes)

    def add_link(self, source, target, link_type="related", weight=1.0):
        """Add a bi-directional link between two concepts."""
        s_key = source.lower().strip()
        t_key = target.lower().strip()

        # Auto-create concepts if they don't exist
        if s_key not in self.nodes:
            self.add_concept(source)
        if t_key not in self.nodes:
            self.add_concept(target)

        # Check if link already exists
        for link in self.edges:
            if link.source == s_key and link.target == t_key:
                link.weight = (link.weight + weight) / 2
                link.last_updated = time.time()
                return link

        # Create new link
        link = ConceptLink(source, target, link_type, weight)
        self.edges.append(link)
        self._adjacency[s_key].add(t_key)
        self._adjacency[t_key].add(s_key)
        self._backlinks_cache[t_key].add(s_key)
        return link

    def get_concept(self, name):
        """Get a concept node by name (case-insensitive)."""
        key = name.lower().strip()
        return self.nodes.get(key)

    def get_links(self, name):
        """Get all outgoing links from a concept."""
        key = name.lower().strip()
        return [e for e in self.edges if e.source == key]

    def get_backlinks(self, name):
        """Get all concepts that link TO this concept (backlinks)."""
        key = name.lower().strip()
        result = []
        for e in self.edges:
            if e.target == key:
                source_node = self.nodes.get(e.source)
                if source_node:
                    result.append((source_node.name, e.link_type, e.weight))
        return result

    def get_neighbors(self, name):
        """Get all directly connected concepts (both directions)."""
        key = name.lower().strip()
        neighbors = set()
        for e in self.edges:
            if e.source == key:
                neighbors.add(e.target)
            if e.target == key:
                neighbors.add(e.source)
        result = []
        for n_key in neighbors:
            node = self.nodes.get(n_key)
            if node:
                result.append(node.name)
        return result

    # ---- [[Wiki-Link]] Integration ----

    def process_text(self, text, source_concept=None):
        """
        Process text for [[wiki-links]] and auto-extract concepts.
        Links: [[Target]] creates a link from source_concept to Target.
        If no source_concept, treats the first [[concept]] as source.
        """
        # Step 1: Extract explicit [[wiki-links]]
        wiki_links = WikiLinkParser.extract(text)

        if wiki_links:
            # Use the first wiki-link as the source concept if none provided
            actual_source = source_concept or wiki_links[0]
            for target in wiki_links:
                if target != actual_source:
                    self.add_concept(target)
                    self.add_link(actual_source, target)
            return wiki_links

        # Step 2: No [[wiki-links]] — auto-extract concepts
        extracted = ConceptExtractor.extract(text)
        if source_concept:
            for concept_name in extracted:
                if concept_name.lower() != source_concept.lower():
                    self.add_concept(concept_name)
                    self.add_link(source_concept, concept_name, weight=min(1.0, extracted[concept_name] * 0.3))
        return list(extracted.keys())

    # ---- Search ----

    def search(self, query, max_results=20):
        """Fuzzy search for concepts matching query."""
        query = query.lower().strip()
        results = []

        for key, node in self.nodes.items():
            score = 0
            # Exact match
            if key == query:
                score = 100
            # Starts with query
            elif key.startswith(query):
                score = 80
            # Query is substring
            elif query in key:
                score = 60
            # Word starts with query
            elif any(w.startswith(query) for w in key.split()):
                score = 40
            # Alias match
            elif any(query in a.lower() for a in node.aliases):
                score = 20

            if score > 0:
                results.append((node, score))

        results.sort(key=lambda x: (-x[1], -x[0].reference_count))
        return [node for node, _ in results[:max_results]]

    def search_aliases(self, query):
        """Search specifically in aliases."""
        query = query.lower()
        results = []
        for key, node in self.nodes.items():
            for alias in node.aliases:
                if query in alias.lower():
                    results.append((node, alias))
                    break
        return results

    # ---- Graph Traversal ----

    def shortest_path(self, start, end, max_depth=10):
        """BFS shortest path between two concepts."""
        s_key = start.lower().strip()
        e_key = end.lower().strip()

        if s_key not in self.nodes or e_key not in self.nodes:
            return None

        visited = {s_key}
        queue = deque([(s_key, [s_key])])

        while queue:
            current, path = queue.popleft()
            if len(path) > max_depth:
                continue

            for neighbor in self._adjacency[current]:
                if neighbor == e_key:
                    full_path = path + [neighbor]
                    return [self.nodes[n].name for n in full_path]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def subgraph(self, center, depth=2):
        """Get all nodes within `depth` hops of `center`."""
        key = center.lower().strip()
        if key not in self.nodes:
            return set()

        result = {key}
        frontier = {key}

        for _ in range(depth):
            next_frontier = set()
            for f in frontier:
                for neighbor in self._adjacency[f]:
                    if neighbor not in result:
                        result.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier

        return {self.nodes[n].name for n in result if n in self.nodes}

    # ---- Statistics ----

    def stats(self):
        """Return memory statistics."""
        if not self.nodes:
            return "Memory graph is empty."

        # Top referenced
        sorted_nodes = sorted(self.nodes.values(), key=lambda n: -n.reference_count)

        # Recent
        recent = sorted(self.nodes.values(), key=lambda n: -n.last_seen)[:5]

        # Link type distribution
        link_types = defaultdict(int)
        for e in self.edges:
            link_types[e.link_type] += 1

        # Components
        from collections import Counter
        neighbor_counts = Counter()
        for n_key in self.nodes:
            neighbor_counts[len(self._adjacency.get(n_key, set()))] += 1

        lines = []
        lines.append(f"Concepts: {len(self.nodes)}")
        lines.append(f"Links: {len(self.edges)}")
        lines.append(f"Link types: {dict(link_types)}")
        lines.append(f"Orphans (0 connections): {neighbor_counts.get(0, 0)}")
        lines.append(f"Most referenced: {', '.join(n.name[:25] for n in sorted_nodes[:5])}")
        lines.append(f"Most recent: {', '.join(n.name[:25] for n in recent)}")

        # Graph density
        if len(self.nodes) > 1:
            max_edges = len(self.nodes) * (len(self.nodes) - 1) / 2
            density = len(self.edges) / max_edges * 100 if max_edges > 0 else 0
            lines.append(f"Graph density: {density:.2f}%")

        return '\n'.join(lines)

    # ---- Persistence ----

    def save(self):
        """Save memory graph to JSON file."""
        data = {
            'nodes': {k: v.to_dict() for k, v in self.nodes.items()},
            'edges': [e.to_dict() for e in self.edges],
            'updated': time.time()
        }
        try:
            with open(self.filepath, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"  [MEMORY] Save error: {e}")
            return False

    def load(self):
        """Load memory graph from JSON file."""
        if not os.path.exists(self.filepath):
            return False
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
            self.nodes = {k: ConceptNode.from_dict(v) for k, v in data.get('nodes', {}).items()}
            self.edges = [ConceptLink.from_dict(e) for e in data.get('edges', [])]
            # Rebuild adjacency
            self._adjacency.clear()
            self._backlinks_cache.clear()
            for e in self.edges:
                self._adjacency[e.source].add(e.target)
                self._adjacency[e.target].add(e.source)
                self._backlinks_cache[e.target].add(e.source)
            return True
        except Exception as e:
            print(f"  [MEMORY] Load error: {e}")
            return False


def fmt_ts(t):
    """Format timestamp for display."""
    try:
        return datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M')
    except:
        return '?'


def demo_memory_graph():
    """Demonstrate the memory graph."""
    print("=" * 60)
    print("MEMORY GRAPH DEMONSTRATION")
    print("=" * 60)

    mg = MemoryGraph(filepath="memory_graph_demo.json")

    # Add seed concepts
    mg.add_concept("Biology", notes="The study of living organisms")
    mg.add_concept("Cell", notes="Basic unit of life")
    mg.add_concept("Mitochondria", notes="Powerhouse of the cell, generates ATP")
    mg.add_concept("DNA", notes="Deoxyribonucleic acid, genetic blueprint")
    mg.add_concept("Protein", notes="Large biomolecules essential for life")
    mg.add_concept("Evolution", notes="Change in heritable characteristics over generations")

    # Add links
    mg.add_link("Biology", "Cell", "contains")
    mg.add_link("Cell", "Mitochondria", "contains")
    mg.add_link("Cell", "DNA", "contains")
    mg.add_link("DNA", "Protein", "codes_for")
    mg.add_link("Biology", "Evolution", "studies")
    mg.add_link("Evolution", "DNA", "involves")

    print("\n[TEST] Concepts added:")
    for name in ["Biology", "Cell", "Mitochondria", "Python"]:
        node = mg.get_concept(name)
        if node:
            print(f"  {node.name} (refs={node.reference_count})")

    print("\n[TEST] Links from 'Biology':")
    for link in mg.get_links("Biology"):
        target = mg.get_concept(link.target)
        tname = target.name if target else link.target
        print(f"  Biology --[{link.link_type}]--> {tname}")

    print("\n[TEST] Backlinks to 'Cell':")
    for source, ltype, weight in mg.get_backlinks("Cell"):
        print(f"  {source} --[{ltype}]--> Cell")

    print("\n[TEST] Neighbors of 'Cell':")
    print(f"  {mg.get_neighbors('Cell')}")

    print("\n[TEST] Shortest path: Biology -> Protein")
    path = mg.shortest_path("Biology", "Protein")
    print(f"  {' -> '.join(path) if path else 'No path found'}")

    print("\n[TEST] Fuzzy search: 'bio':")
    for node in mg.search("bio"):
        print(f"  {node.name} (refs={node.reference_count})")

    print("\n[TEST] [[Wiki-link]] processing:")
    text = "[[Cell]] biology studies [[Mitochondria]] and [[DNA]]"
    links = mg.process_text(text, source_concept="Cell Biology")
    print(f"  Extracted links: {links}")

    print("\n[TEST] Auto-extraction (no [[wiki-links]]):")
    text2 = "The CRISPR system allows precise gene editing in living organisms."
    concepts = mg.process_text(text2, source_concept="Gene Editing")
    print(f"  Auto-extracted: {concepts[:5]}")

    print("\n[TEST] Statistics:")
    print(mg.stats())

    # Save
    mg.save()
    print("\n  Saved to memory_graph_demo.json")

    # Clean up demo file
    try:
        os.remove("memory_graph_demo.json")
    except:
        pass

    print("\n" + "=" * 60)
    print("MEMORY GRAPH DEMO COMPLETE")
    print("=" * 60)

    return mg


if __name__ == "__main__":
    demo_memory_graph()
