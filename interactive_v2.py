"""
BIOLOGIC LLM V2 — Interactive Shell
BPE | 5120 context | Sliding window | CoT reasoning | Web learning | Safety nets
Memory graph | Episodic memory | ADHD multi-thread | Dream engine | GPU acceleration

Commands:
  teach <text>         - Learn from text, auto-extract concepts
  ask <question>       - Generate a response (human-like gestalt)
  reason <query>       - Chain-of-thought reasoning
  verify <text>        - Self-verify a reasoning trace
  web <topic>          - Learn from Wikipedia in real-time
  sleep                - Consolidation + dream cycle
  self                 - Self-improvement analysis
  state                - System state
  memory               - Knowledge graph overview
  memory search <q>    - Search concepts
  memory links <c>     - Show connections
  memory backlinks <c> - Show backlinks
  memory path <a> <b>  - Shortest path
  memory graph <c>     - Show subgraph
  memory stats         - Graph statistics
  remember <cue>       - Pattern-complete a memory from a few words
  associate <cue>      - Fill in missing details from partial cues
  dream                - Daydream right now
  adhd                 - Toggle ADHD multi-thread mode
  help                 - Show this help
  exit/quit            - Shut down
"""

import sys
import os
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokenizer import BPETokenizer
from safety_nets import SafetySystem
from reasoning_engine import ReasoningEngine
from web_learner import WebLearner
from memory_graph import MemoryGraph


def print_header():
    print()
    print("=" * 60)
    print("  BIOLOGIC LLM V2")
    print("  BPE | 5120 Context | Sliding Window Attention | GPU")
    print("  CoT Reasoning | Web Learning | 7 Safety Nets")
    print("  Memory Graph | Episodic Memory | ADHD Generation | Dreams")
    print("=" * 60)
    print()


def safe_print(text):
    safe = text.encode('ascii', errors='replace').decode('ascii')
    print(safe)


def learn_encoded(model, encoded, value_label=0.5, task_type="general"):
    """Learn from encoded tokens with adaptive chunk size."""
    raw_len = len(encoded)
    if raw_len < 4:
        return 0
    chunk_size = min(32, max(4, raw_len - 2))
    count = 0
    for i in range(0, max(1, raw_len - chunk_size - 1), max(chunk_size // 2, 1)):
        chunk = encoded[i:i + chunk_size]
        target = encoded[i + 1:i + chunk_size + 1]
        if len(chunk) == len(target) and len(chunk) > 1:
            model.learn_from_interaction(chunk, target, value_label=value_label, task_type=task_type)
            count += 1
    return count


def is_gpu_available():
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False


def main():
    print_header()

    # ============================================================
    # 1. BPE TOKENIZER
    # ============================================================
    print("[1/6] Loading BPE tokenizer...")
    tokenizer_path = "bpe_vocab.json"
    tokenizer = BPETokenizer(vocab_size=4096)
    if os.path.exists(tokenizer_path):
        tokenizer.load(tokenizer_path)
        print(f"      Vocabulary: {tokenizer.get_vocab_size()} tokens")
    else:
        print("      No tokenizer found. Running minimal training...")
        tokenizer.train("Hello world. BPE tokenizer for Biologic LLM V2.")
        tokenizer.save(tokenizer_path)
        print(f"      Vocabulary: {tokenizer.get_vocab_size()} tokens")

    # ============================================================
    # 2. V2 MODEL
    # ============================================================
    print("[2/6] Loading V2 model...")
    from biologic_v2 import BiologicLLMV2, create_model, tokenizer as v2_tokenizer, DEVICE
    if v2_tokenizer.get_vocab_size() > tokenizer.get_vocab_size():
        tokenizer = v2_tokenizer

    # Auto-scale on GPU, use defaults on CPU
    is_gpu = str(DEVICE) != 'cpu' and is_gpu_available()
    embed_dim = 512 if is_gpu else 256
    num_layers = 12 if is_gpu else 8

    model = create_model(
        vocab_size=tokenizer.get_vocab_size(),
        embed_dim=embed_dim,
        num_layers=num_layers,
        do_seed_learning=True,
        tokenizer_ref=tokenizer,
        auto_scale=True
    )
    print(f"      {sum(p.numel() for p in model.parameters()):,} parameters on {DEVICE}")
    print(f"      Context: {model.max_context} tokens | Window: {model.window_size}")

    # ============================================================
    # 3. SAFETY SYSTEM
    # ============================================================
    print("[3/6] Initializing safety system...")
    safety = SafetySystem()
    print(f"      7 safety nets active")

    # ============================================================
    # 4. REASONING ENGINE
    # ============================================================
    print("[4/6] Initializing reasoning engine...")
    reasoning = ReasoningEngine(model, tokenizer)
    print(f"      Chain-of-thought + self-verification ready")

    # ============================================================
    # 5. WEB LEARNER
    # ============================================================
    print("[5/6] Initializing web learner...")
    web_learner = WebLearner(safety_system=safety)
    print(f"      Source: Wikipedia API")

    # ============================================================
    # 6. MEMORY GRAPH (Obsidian-like)
    # ============================================================
    print("[6/6] Loading knowledge graph...")
    memory_graph = MemoryGraph(filepath="memory_graph.json")
    print(f"      {len(memory_graph.nodes)} concepts, {len(memory_graph.edges)} links")

    # ============================================================
    # 7. EPISODIC MEMORY
    # ============================================================
    from episodic_memory import EpisodicMemory
    episodic = EpisodicMemory(filepath="episodic_memory.json")
    print(f"      {len(episodic.traces)} episodic traces")

    # ============================================================
    # 8. ADHD GENERATOR
    # ============================================================
    from adhd_thought import ADHDGenerator
    adhd_gen = ADHDGenerator()
    print(f"      ADHD multi-thread generator ready")
    adhd_enabled = False

    # ============================================================
    # 9. DREAM ENGINE
    # ============================================================
    from dream_engine import DreamEngine
    dreams = DreamEngine(episodic_memory=episodic, memory_graph=memory_graph)
    print(f"      Dream engine ready")
    print()

    print("System is ready. Type 'help' for commands or 'exit' to quit.")
    print()

    interaction_count = 0

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print("Shutting down...")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        # === EXIT ===
        if cmd in ['exit', 'quit', 'stop']:
            print("\n  [SLEEP] Final consolidation...")
            model.consolidate_memory()
            print("  [DREAM] Dream cycle...")
            dream_result = dreams.dream_cycle(model, tokenizer, n_dreams=2)
            if dream_result.get('dreams'):
                print(f"  [DREAM] {len(dream_result['dreams'])} dreams, {len(dream_result['new_links'])} new links")
                dreams.learn_from_dreams(model, tokenizer, dream_result['dreams'])
            model.self_improve()
            memory_graph.save()
            episodic.save()
            print("  [MEMORY] Knowledge graph saved.")
            print("  [EPISODIC] Episodic memory saved.")

            print("\n" + "=" * 60)
            print("SESSION SUMMARY")
            print("=" * 60)
            state = model.get_state_summary()
            for key, val in state.items():
                print(f"  {key}: {val}")
            print(f"  Interaction count: {interaction_count}")
            print(f"  Memory concepts: {len(memory_graph.nodes)}")
            print(f"  Memory links: {len(memory_graph.edges)}")
            print(f"  Episodic traces: {len(episodic.traces)}")
            print(f"  Dreams had: {dreams.dream_count}")
            print(f"  ADHD mode: {'ON' if adhd_enabled else 'OFF'}")
            print()
            break

        # === HELP ===
        if cmd == 'help':
            print("Commands:")
            print("  teach <text>          - Learn from text, extract concepts")
            print("  ask <question>        - Generate a human-like response")
            print("  reason <query>        - Chain-of-thought reasoning")
            print("  verify <text>         - Self-verify reasoning trace")
            print("  web <topic>           - Learn from Wikipedia")
            print("  sleep                 - Consolidation + dream cycle")
            print("  self                  - Self-improvement analysis")
            print("  state                 - System state overview")
            print("  memory                - Knowledge graph overview")
            print("  memory search <q>     - Search concepts")
            print("  memory links <c>      - Show connections")
            print("  memory backlinks <c>  - Show backlinks")
            print("  memory path <a> <b>   - Shortest path")
            print("  memory graph <c>      - Show subgraph")
            print("  memory stats          - Graph statistics")
            print("  remember <cue>        - Reconstruct a memory from a few words")
            print("  associate <cue>       - Pattern complete from partial cues")
            print("  dream                 - Daydream right now")
            print("  adhd                  - Toggle ADHD multi-thread mode")
            print("  train [steps]         - Train on Nemotron dataset")
            print("  help                  - This help")
            print("  exit/quit             - Shut down")
            continue

        # === DATASET TRAINING ===
        if cmd.startswith('train'):
            parts = cmd.split()
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 200
            print(f"  [TRAIN] Loading Nemotron dataset ({n} examples)...")
            try:
                from datasets import load_dataset
                ds = load_dataset("nvidia/Nemotron-Cascade-2-SFT-Data", "chat", split="train", streaming=True)
                ds_iter = iter(ds)
                model.train()
                chunk_size = 48; stride = 24
                step_count = 0; nan_count = 0
                for ex_idx in range(n):
                    try:
                        example = next(ds_iter)
                    except StopIteration:
                        break
                    msg = example.get('messages', [])
                    texts = []
                    for m in msg:
                        c = m.get('content', '')
                        if isinstance(c, str): texts.append(c)
                        elif isinstance(c, list): texts.extend([x.get('text','') for x in c if isinstance(x, dict) and x.get('text')])
                    text = ' '.join(texts)
                    if len(text) < 20: continue
                    encoded = tokenizer.encode(text)
                    for sp in range(0, len(encoded)-chunk_size-1, stride):
                        ck = encoded[sp:sp+chunk_size]; tg = encoded[sp+1:sp+chunk_size+1]
                        if len(ck) != len(tg) or len(ck) < 2: continue
                        result = model.learn_from_interaction(ck, tg, value_label=0.5, task_type='chat')
                        if result['loss'] and not np.isnan(result['loss']):
                            step_count += 1
                        else:
                            nan_count += 1
                print(f"  [TRAIN] Done: {step_count} steps, {nan_count} NaN")
                model.consolidate_memory()
            except ImportError:
                print("  [TRAIN] Need 'datasets' library: pip install datasets")
            except Exception as e:
                print(f"  [TRAIN] Error: {e}")
            continue

        # === ADHD TOGGLE ===
        if cmd == 'adhd':
            adhd_enabled = adhd_gen.toggle_adhd()
            print(f"  [ADHD] Multi-thread generation: {'ON' if adhd_enabled else 'OFF'}")
            continue

        # === STATE ===
        if cmd == 'state':
            state = model.get_state_summary()
            print("-" * 50)
            print("SYSTEM STATE")
            print("-" * 50)
            for key, val in state.items():
                print(f"  {key}: {val}")
            print(f"  safety_rejected: {safety.total_inputs_rejected}")
            print(f"  safety_accepted: {safety.total_inputs_accepted}")
            print(f"  web_topics: {len(web_learner.learned_topics)}")
            print(f"  reasoning_history: {len(reasoning.reasoning_history)}")
            print(f"  memory_concepts: {len(memory_graph.nodes)}")
            print(f"  memory_links: {len(memory_graph.edges)}")
            print(f"  episodic_traces: {len(episodic.traces)}")
            print(f"  dreams: {dreams.dream_count}")
            print(f"  adhd_mode: {'ON' if adhd_enabled else 'OFF'}")
            print(f"  device: {state.get('device', '?')}")
            if hasattr(model, 'get_task_summary'):
                ts = model.get_task_summary()
                if ts.strip():
                    print(f"  --- Task Profiles ---")
                    print(ts)
            if state.get('meta_base_range'):
                print(f"  --- Adaptive Plasticity ---")
                print(f"  meta_base: {state['meta_base_range']}")
                print(f"  avg_gate: {state['avg_gate']}")
                print(f"  surprise_scale: {state['surprise_scale_range']}")
            continue

        # === MEMORY (working memory + memory graph) ===
        if cmd == 'memory':
            rsummary = reasoning.get_memory_summary()
            print("--- Working Memory (Reasoning) ---")
            print(f"  Thoughts: {len(rsummary['thoughts'])}")
            for t in rsummary['thoughts'][-5:]:
                safe_print(f"    - {t[:100]}")
            print(f"  Facts: {len(rsummary['facts'])}")
            print()
            print("--- Knowledge Graph ---")
            print(f"  {len(memory_graph.nodes)} concepts, {len(memory_graph.edges)} links")
            print(f"--- Episodic Memory ---")
            print(f"  {len(episodic.traces)} traces")
            print(f"  Type 'memory stats' for graph details, 'remember <cue>' for episodes.")
            continue

        # === MEMORY SUB-COMMANDS ===
        if cmd.startswith('memory '):
            sub = user_input[7:].strip()
            if not sub or sub == 'stats':
                print(memory_graph.stats())
                print(episodic.stats())
                continue

            if sub.startswith('search '):
                query = sub[7:]
                results = memory_graph.search(query)
                if results:
                    print(f"  Found {len(results)} concepts matching '{query}':")
                    for node in results[:15]:
                        neighbors = memory_graph.get_neighbors(node.name)
                        nstr = ', '.join(neighbors[:5])
                        safe_print(f"    {node.name} (refs={node.reference_count}) -> {nstr}")
                else:
                    print(f"  No concepts found matching '{query}'")
                continue

            if sub.startswith('links '):
                concept = sub[6:]
                node = memory_graph.get_concept(concept)
                if not node:
                    print(f"  Concept '{concept}' not found in memory.")
                    print(f"  Try 'memory search {concept}' to find similar concepts.")
                    continue
                print(f"  Links from '{node.name}':")
                links = memory_graph.get_links(concept)
                if links:
                    for link in links:
                        target = memory_graph.get_concept(link.target)
                        tname = target.name if target else link.target
                        safe_print(f"    --[{link.link_type}]--> {tname} (w={link.weight:.2f})")
                else:
                    print(f"    (no outgoing links)")
                print(f"  Backlinks ({len(memory_graph.get_backlinks(concept))}):")
                for source, ltype, weight in memory_graph.get_backlinks(concept):
                    safe_print(f"    {source} --[{ltype}]--> {node.name}")
                continue

            if sub.startswith('backlinks '):
                concept = sub[10:]
                node = memory_graph.get_concept(concept)
                if not node:
                    print(f"  Concept '{concept}' not found.")
                    continue
                backlinks = memory_graph.get_backlinks(concept)
                if backlinks:
                    print(f"  Concepts linking TO '{node.name}':")
                    for source, ltype, weight in backlinks:
                        safe_print(f"    {source} --[{ltype}]-->")
                else:
                    print(f"  Nothing links to '{node.name}'")
                continue

            if sub.startswith('path '):
                parts = sub[5:].split(None, 1)
                if len(parts) < 2:
                    print("  Usage: memory path <concept_a> <concept_b>")
                    continue
                a, b = parts[0], parts[1]
                path = memory_graph.shortest_path(a, b)
                if path:
                    print(f"  Path: {' -> '.join(path)}")
                else:
                    print(f"  No path found between '{a}' and '{b}'")
                continue

            if sub.startswith('graph '):
                concept = sub[6:]
                subgraph = memory_graph.subgraph(concept)
                if subgraph:
                    node_list = sorted(subgraph)
                    print(f"  Subgraph around '{concept}' ({len(node_list)} concepts):")
                    for n in node_list:
                        links = memory_graph.get_links(n)
                        targets = []
                        for link in links:
                            t = memory_graph.get_concept(link.target)
                            if t and t.name in subgraph:
                                targets.append(t.name)
                        if targets:
                            safe_print(f"    {n} -> {', '.join(targets[:5])}")
                        else:
                            safe_print(f"    {n}")
                else:
                    print(f"  Concept '{concept}' not found.")
                continue

            print("  Unknown memory command. Try: search, links, backlinks, path, graph, stats")
            continue

        # === REMEMBER (episodic reconstruction) ===
        if cmd.startswith('remember '):
            cue = user_input[9:]
            if not cue:
                print("  Usage: remember <cue>")
                continue

            result = episodic.reconstruct(cue, model, tokenizer, temperature=0.85)
            if result:
                safe_print(f"  [REMEMBER] Anchor: {', '.join(result['anchor'][:3])}")
                safe_print(f"  Valence: {result['valence']:.2f}")
                safe_print(f"  Memory: {result['reconstruction'][:400]}")
                if result['associated']:
                    safe_print(result['associated'])
            else:
                print(f"  [REMEMBER] No matching memories found for '{cue}'")
                print(f"  Try learning something first with 'teach' or 'web'.")
            continue

        # === ASSOCIATE (pattern completion) ===
        if cmd.startswith('associate '):
            cue = user_input[10:]
            if not cue:
                print("  Usage: associate <partial_cues>")
                continue

            result = episodic.pattern_complete(cue, model, tokenizer, temperature=0.9)
            if result:
                safe_print(f"  [ASSOCIATE] Cues: {result['cues']}")
                safe_print(f"  Matched: {', '.join(result['matched_anchor'][:3])}")
                safe_print(f"  Completed: {result['completed'][:400]}")
            else:
                print(f"  [ASSOCIATE] No completion for '{cue}'")
            continue

        # === DREAM ===
        if cmd == 'dream':
            print("  [DREAM] Daydreaming...")
            result = dreams.dream(model, tokenizer, dream_type="remix", temperature=1.0)
            if result:
                safe_print(f"\n  Dream: {result['dream_text'][:400]}")
                if result['new_links']:
                    safe_print(f"  New links: {', '.join(result['new_links'])}")
                # Learn from daydream
                count = dreams.learn_from_dreams(model, tokenizer, [result])
                print(f"  [DREAM] Learned {count} steps from daydream")
            else:
                print("  [DREAM] Not enough memories to dream yet.")
            continue

        # === SLEEP ===
        if cmd == 'sleep':
            model.consolidate_memory()
            print("  [SLEEP] Running dream cycle...")
            dream_result = dreams.dream_cycle(model, tokenizer, n_dreams=2)
            if dream_result.get('dreams'):
                for d in dream_result['dreams']:
                    if d:
                        safe_print(f"    Dream ({d['type']}): {d['dream_text'][:80]}...")
                        if d.get('new_links'):
                            safe_print(f"    Links: {', '.join(d['new_links'][:3])}")
                dream_learn_count = dreams.learn_from_dreams(model, tokenizer, dream_result['dreams'])
                print(f"  [DREAM] Learned {dream_learn_count} steps from {len(dream_result['dreams'])} dreams")
            else:
                print("  [DREAM] Not enough memories to dream. Keep learning!")
            memory_graph.save()
            episodic.save()
            print("  [SLEEP] Cycle complete.")
            continue

        # === SELF ===
        if cmd == 'self':
            assessment = model.self_improve()
            if assessment:
                safe_print(f"  Surprise: {assessment['avg_surprise']:.3f}")
                safe_print(f"  Loss: {assessment['avg_loss']:.3f}")
                safe_print(f"  Curiosity: {assessment['curiosity']:.2f}")
                if assessment['suggestions']:
                    safe_print(f"  Suggestions: {', '.join(assessment['suggestions'])}")
            else:
                print("  Not enough experience yet (need 10+ interactions).")
            continue

        # === TEACH ===
        if cmd.startswith('teach '):
            teach_text = user_input[6:]
            if not teach_text:
                print("  Usage: teach <text>")
                continue

            allowed, reason, details = safety.pre_check(teach_text, source='user')
            if not allowed:
                safe_print(f"  [SAFETY] Rejected: {reason}")
                continue

            encoded = tokenizer.encode(teach_text)
            learn_count = learn_encoded(model, encoded, 0.5, task_type="teach")

            safety.post_learn(teach_text)
            interaction_count += 1
            print(f"  [LEARN] Absorbed {len(teach_text)} chars in {learn_count} steps")

            # Extract concepts into memory_graph
            concepts = memory_graph.process_text(teach_text, source_concept="User Input")
            if concepts:
                print(f"  [MEMORY] Linked {len(concepts)} concepts")

            # Store in episodic memory
            episodic.store(teach_text, concepts=concepts, valence=0.5, source="teach")
            print(f"  [EPISODIC] Stored as memory trace")

            memory_graph.save()
            episodic.save()

            if model.total_experience % 15 == 0:
                model.consolidate_memory()
            continue

        # === REASON ===
        if cmd.startswith('reason '):
            query = user_input[7:]
            if not query:
                print("  Usage: reason <query>")
                continue

            if adhd_enabled:
                print("  [REASON] ADHD mode — multi-perspective reasoning...")
                result = adhd_gen.reason_with_adhd(
                    query, model, tokenizer, max_tokens=400
                )
                print()
                safe_print(f"  Question: {query}")
                print(f"  Phases: {', '.join(result['phases'])}")
                print()
                safe_print(f"  Output:")
                print(f"  {'-'*40}")
                safe_print(f"  {result['full_output'][:800]}")
                print(f"  {'-'*40}")

                # Store reasoning in episodic memory
                episodic.store(
                    result['full_output'],
                    concepts=[w for w in query.split()[:5]],
                    valence=0.4,
                    source="reason"
                )
            else:
                result = reasoning.reason(query)

                print()
                safe_print(f"  Question: {result['question']}")
                print(f"  Type: {result['type']}")
                if result['sub_questions']:
                    safe_print(f"  Sub-questions: {[sq['question'] for sq in result['sub_questions']]}")
                print(f"  Steps: {len(result['reasoning_steps'])}")
                print()
                safe_print(f"  Trace:")
                print(f"  {'-'*40}")
                safe_print(f"  {result['reasoning_trace'][:600]}")
                print(f"  {'-'*40}")
                print()
                verify = result['verification']
                status = 'PASS' if verify['passed'] else 'NEEDS REVIEW'
                print(f"  Self-verification: {status} (score: {verify['score']:.2f})")
                if verify['details']:
                    for d in verify['details']:
                        safe_print(f"    Issue: {d}")

            interaction_count += 1
            continue

        # === VERIFY ===
        if user_input.lower().startswith('verify '):
            trace = user_input[7:]
            if not trace:
                print("  Usage: verify <reasoning trace text>")
                continue

            result = reasoning.verify(trace)
            status = 'PASS' if result['passed'] else 'NEEDS REVIEW'
            print(f"  Verification: {status} (score: {result['score']:.2f})")
            for check in result['checks']:
                icon = '+' if check['passed'] else '-'
                print(f"    [{icon}] {check['message']}")
            continue

        # === WEB ===
        if cmd.startswith('web '):
            topic = user_input[4:]
            if not topic:
                print("  Usage: web <topic>")
                continue

            print(f"  [WEB] Learning about '{topic}'...")
            result = web_learner.learn(topic, max_chars=2000)

            if result['success']:
                content = result['content'][:500]
                safe_print(f"  Source: {result['source']}")
                safe_print(f"  Content ({len(result['content'])} chars): {content}...")

                encoded = tokenizer.encode(result['content'])
                learn_encoded(model, encoded, 0.6, task_type=f"web:{topic}")

                safety.post_learn(result['content'])
                print(f"  [WEB] Learned. Experiences: {model.total_experience}")

                # Extract concepts into memory_graph
                source_name = result.get('source', topic)
                concepts = memory_graph.process_text(result['content'], source_concept=source_name)
                if concepts:
                    print(f"  [MEMORY] Linked {len(concepts)} concepts from '{source_name}'")

                # Store in episodic memory
                episodic.store(
                    result['content'],
                    concepts=concepts or [topic],
                    valence=0.6,
                    source=f"web:{source_name}"
                )
                print(f"  [EPISODIC] Stored memory trace")

                memory_graph.save()
                episodic.save()
            else:
                safe_print(f"  [WEB] Failed: {result.get('error', 'unknown')}")

            interaction_count += 1
            continue

        # === ASK ===
        if '?' in user_input or cmd.startswith('ask '):
            question = user_input[4:] if cmd.startswith('ask ') else user_input
            safe_print(f"  [THINK] '{question}'")

            if adhd_enabled:
                # ADHD multi-thread generation
                output = adhd_gen.generate(
                    question, model, tokenizer, max_tokens=150,
                    temperature=0.8
                )
                safe_print(f"  {output[:500]}")
            else:
                # Human-like gestalt generation
                prompt_ids = tokenizer.encode(question)
                generated_ids = model.generate_human(
                    prompt_ids, max_new_tokens=150,
                    gestalt_temp=1.5, main_temp=0.8
                )
                response = tokenizer.decode(generated_ids)
                safe_print(f"  {response[:400]}")

            interaction_count += 1

            combined = tokenizer.encode(question + (response if not adhd_enabled else output)[:200])
            learn_encoded(model, combined, 0.3, task_type="ask")
            continue

        # === DEFAULT: teach and generate ===
        safe_print(f"  [LEARN] Processing input...")

        allowed, reason, details = safety.pre_check(user_input, source='user')
        if not allowed:
            safe_print(f"  [SAFETY] Rejected: {reason}")
            continue

        encoded = tokenizer.encode(user_input)
        learn_encoded(model, encoded, 0.4, task_type="general")

        safety.post_learn(user_input)
        interaction_count += 1

        # Extract concepts
        concepts = memory_graph.process_text(user_input, source_concept="User Input")
        if concepts:
            print(f"  [MEMORY] Linked {len(concepts)} concepts")

        # Store in episodic memory
        episodic.store(user_input, concepts=concepts, valence=0.4, source="default")
        memory_graph.save()
        episodic.save()

        prompt_ids = tokenizer.encode(user_input[:50])
        generated_ids = model.generate_human(
            prompt_ids, max_new_tokens=80,
            gestalt_temp=1.4, main_temp=0.8
        )
        response = tokenizer.decode(generated_ids)
        safe_print(f"  {response[:200]}")
        print(f"  [Experiences: {model.total_experience}]")

        if model.total_experience % 15 == 0:
            model.consolidate_memory()
        if model.total_experience % 30 == 0:
            model.self_improve()

    print()
    print("Biologic LLM V2 shut down. It remembers everything you taught it.")
    print(f"Knowledge graph: {len(memory_graph.nodes)} concepts, {len(memory_graph.edges)} links")
    print(f"Episodic memory: {len(episodic.traces)} traces")
    print(f"Dreams: {dreams.dream_count}")


if __name__ == "__main__":
    main()
