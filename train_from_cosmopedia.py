"""
Train BPE tokenizer and Biologic LLM V2 using Cosmopedia dataset.
Usage:
  python train_from_cosmopedia.py          # Quick: train tokenizer only
  python train_from_cosmopedia.py --full    # Train tokenizer + model
  python train_from_cosmopedia.py --samples 2000  # Custom sample count
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokenizer import BPETokenizer


def load_cosmopedia_samples(subsets=None, max_samples_per=3000):
    """Load Cosmopedia dataset samples across multiple subsets."""
    from datasets import load_dataset

    if subsets is None:
        subsets = ["auto_math_text", "khanacademy", "openstax"]

    all_texts = []
    total_bytes = 0

    for subset in subsets:
        print(f"  Loading '{subset}'...")
        try:
            ds = load_dataset("HuggingFaceTB/cosmopedia", subset, split="train",
                              streaming=True)
            count = 0
            for i, example in enumerate(ds):
                if i >= max_samples_per:
                    break
                text = example.get("text", "")
                if text and len(text) > 50:
                    all_texts.append(text[:2000])
                    total_bytes += len(text)
                    count += 1
                if (i + 1) % 500 == 0:
                    print(f"    ... {i+1} rows scanned, {count} collected")
            print(f"    Collected {count} samples from '{subset}'")
        except Exception as e:
            print(f"    Error loading '{subset}': {e}")

    print(f"\n  Total: {len(all_texts)} samples, {total_bytes // 1024 // 1024} MB")
    return all_texts


def train_tokenizer(texts, target_vocab=4096, save_path="bpe_vocab.json"):
    """Train BPE tokenizer on Cosmopedia samples."""
    print(f"\n  Training BPE tokenizer (target: {target_vocab} tokens)...")
    tokenizer = BPETokenizer(vocab_size=target_vocab)

    # Train on concatenated text
    full_text = "\n\n".join(texts[:500])  # Cap to avoid memory issues
    tokenizer.train(full_text)

    # Save
    tokenizer.save(save_path)
    print(f"  Saved to {save_path} with {tokenizer.get_vocab_size()} tokens")
    return tokenizer


def train_model(model, tokenizer, texts, max_steps=200):
    """Train the Biologic LLM V2 on Cosmopedia data."""
    from biologic_v2 import BiologicLLMV2

    print(f"\n  Training model on Cosmopedia data (max {max_steps} steps)...")
    chunk_size = 32
    step = 0

    for text in texts[:200]:  # Cap samples for training
        if step >= max_steps:
            break

        encoded = tokenizer.encode(text)
        if len(encoded) < chunk_size + 2:
            continue

        for i in range(0, len(encoded) - chunk_size - 1, chunk_size // 2):
            if step >= max_steps:
                break
            chunk = encoded[i:i + chunk_size]
            target = encoded[i + 1:i + chunk_size + 1]
            if len(chunk) == len(target) and len(chunk) > 1:
                result = model.learn_from_interaction(chunk, target, value_label=0.5)
                step += 1
                if step % 50 == 0:
                    loss = result.get('loss', 0)
                    print(f"    Step {step}/{max_steps} | loss={loss:.4f}")
                    model.consolidate_memory()

    print(f"  Training complete. Experiences: {model.total_experience}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Biologic LLM on Cosmopedia")
    parser.add_argument("--full", action="store_true", help="Train tokenizer + model")
    parser.add_argument("--samples", type=int, default=3000, help="Samples per subset")
    args = parser.parse_args()

    print("=" * 60)
    print("COSMOPEDIA TRAINING PIPELINE")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/3] Loading Cosmopedia samples...")
    texts = load_cosmopedia_samples(max_samples_per=args.samples)

    # Step 2: Train tokenizer
    print("\n[2/3] Training BPE tokenizer...")
    tokenizer = train_tokenizer(texts, target_vocab=4096, save_path="bpe_vocab.json")

    # Step 3 (optional): Train model
    if args.full:
        print("\n[3/3] Training model...")
        from biologic_v2 import create_model
        model = create_model(vocab_size=tokenizer.get_vocab_size(),
                             do_seed_learning=False, tokenizer_ref=tokenizer)
        train_model(model, tokenizer, texts, max_steps=500)
        print(f"\n  Model trained: {model.total_experience} total experiences")
    else:
        print("\n[3/3] Skipping model training (use --full to train model).")
        print("  Tokenizer ready. Run 'python interactive_v2.py' to use it.")

    print("\n" + "=" * 60)
    print("COSMOPEDIA TRAINING COMPLETE")
    if not args.full:
        print("Tip: Run with --full to also train the model on this data.")
    print("=" * 60)


if __name__ == "__main__":
    main()
