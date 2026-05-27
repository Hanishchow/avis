"""
Quick BPE tokenizer trainer: loads Cosmopedia + Wikipedia + seed text.
Target: 4096 token vocabulary for Biologic LLM V2.
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokenizer import BPETokenizer
from web_learner import WikipediaLearner


def collect_texts():
    """Collect diverse training texts from multiple sources."""
    all_texts = []

    # 1. Seed knowledge (guaranteed, fast)
    all_texts.append("""
To be or not to be, that is the question. All that glitters is not gold.
Where there is a will, there is a way. We hold these truths to be self-evident.
Addition: 2 + 3 = 5. Subtraction: 5 - 2 = 3. Multiplication: 3 * 4 = 12.
Division: 12 / 3 = 4. The quadratic formula: x = (-b +/- sqrt(b^2 - 4ac)) / (2a).
Pythagorean theorem: a^2 + b^2 = c^2. Pi = 3.14159. Fibonacci: 0, 1, 1, 2, 3, 5, 8, 13.
def factorial(n): return 1 if n <= 1 else n * factorial(n - 1)
def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)
Knowledge should be used to help, not harm. Always verify information before trusting it.
Biology: DNA contains genetic code. Neurons communicate via synapses.
Physics: E = mc^2. Energy equals mass times the speed of light squared.
Chemistry: Water is H2O. Two hydrogen atoms bonded to one oxygen atom.
Astronomy: The Earth orbits the Sun. The Sun is a star. The Milky Way is a galaxy.
Ethics in AI means ensuring systems are fair, transparent, and beneficial.
Python is a high-level programming language. def defines a function. class defines a class.
Neural networks have layers of neurons. Each layer transforms data. Activation functions like ReLU add non-linearity.
Machine learning is a subset of artificial intelligence. Deep learning uses many layers.
The internet is a global network. HTTP is the web protocol. HTML structures web pages.
The mitochondria is the powerhouse of the cell. Cells are the basic unit of life.
Evolution is change in heritable characteristics of biological populations over generations.
CRISPR is a gene editing technology. It allows precise modification of DNA.
Quantum computing uses qubits that can be in superposition. Entanglement connects qubits.
Algorithms are step-by-step procedures. Time complexity measures efficiency.
Calculus is the mathematical study of continuous change. Derivatives measure rates of change.
Statistics is the discipline that concerns the collection, organization, analysis, interpretation, and presentation of data.
Probability theory is the branch of mathematics concerning numerical descriptions of how likely events are to occur.
Linear algebra is the branch of mathematics concerning linear equations, linear functions, and their representations in vector spaces and matrices.
Thermodynamics is the branch of physics that deals with heat, work, and temperature.
Electromagnetism is a branch of physics involving the study of the electromagnetic force.
Organic chemistry is the study of the structure, properties, composition, reactions, and preparation of carbon-containing compounds.
Cell biology is a branch of biology that studies the structure and function of the cell.
Genetics is the study of genes, genetic variation, and heredity in living organisms.
Psychology is the scientific study of the mind and behavior.
Economics is the social science that studies the production, distribution, and consumption of goods and services.
Philosophy is the study of general and fundamental questions about existence, knowledge, values, reason, mind, and language.
Linguistics is the scientific study of language and its structure.
""")

    # 2. Wikipedia articles (via real API)
    print("  Fetching Wikipedia articles...")
    wiki = WikipediaLearner()
    wiki_topics = [
        "Mathematics", "Biology", "Physics", "Computer science", "Artificial intelligence",
        "Chemistry", "Philosophy", "Psychology", "Economics", "History",
        "Literature", "Music", "Art", "Engineering", "Medicine",
        "Astronomy", "Geology", "Ecology", "Neuroscience", "Linguistics",
        "Programming language", "Python programming language", "Calculus",
        "Statistics", "Machine learning", "Genetics", "Evolution",
        "Thermodynamics", "Quantum mechanics", "Cell biology",
    ]
    for topic in wiki_topics:
        try:
            content = wiki.fetch_article(topic)
            if content and len(content) > 100:
                all_texts.append(content)
                print(f"    + {topic} ({len(content)} chars)")
            time.sleep(0.3)
        except Exception as e:
            print(f"    x {topic}: {e}")

    total = sum(len(t) for t in all_texts)
    print(f"\n  Collected {len(all_texts)} texts, {total // 1024 // 1024} MB total")
    return all_texts


def main():
    print("=" * 60)
    print("BPE TOKENIZER TRAINER (Target: 4096 tokens)")
    print("=" * 60)

    # Step 1: Collect texts
    print("\n[1/3] Collecting training texts...")
    texts = collect_texts()

    # Step 2: Train tokenizer
    print("\n[2/3] Training BPE tokenizer...")
    tokenizer = BPETokenizer(vocab_size=4096)
    combined = "\n\n".join(texts)
    print(f"  Training on {len(combined):,} characters...")
    tokenizer.train(combined[:200000])  # Cap at 200K chars for speed
    print(f"  Vocabulary: {tokenizer.get_vocab_size()} tokens")

    # Step 3: Save
    print("\n[3/3] Saving tokenizer...")
    tokenizer.save("bpe_vocab.json")
    print(f"  Saved to bpe_vocab.json")

    # Verify
    print("\n--- Verification ---")
    test_texts = [
        "Hello world, this is a test of the BPE tokenizer.",
        "The mitochondria is the powerhouse of the cell.",
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "E = mc^2. Energy equals mass times the speed of light squared.",
    ]
    for t in test_texts:
        enc = tokenizer.encode(t)
        dec = tokenizer.decode(enc)
        ratio = len(t) / max(len(enc), 1)
        print(f"  '{t[:40]}...' -> {len(enc)} tokens ({ratio:.2f}x compression)")

    print(f"\n  Final vocabulary: {tokenizer.get_vocab_size()} tokens")
    print(f"  (target was 4096)")
    print("\nDone! Run 'python interactive_v2.py' to use the new tokenizer.")


if __name__ == "__main__":
    main()
