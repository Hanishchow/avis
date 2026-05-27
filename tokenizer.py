"""
BPE Tokenizer — Byte Pair Encoding from scratch.
Converts text to tokens and back. Trained on seed data.
"""

import re
from collections import defaultdict, Counter

class BPETokenizer:
    """Byte Pair Encoding tokenizer built from scratch."""
    
    SPECIAL_TOKENS = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<BOS>': 2,
        '<EOS>': 3,
        '<MASK>': 4,
        '<SEP>': 5,
        '<CLS>': 6,
        '<REASON>': 7,
        '</REASON>': 8,
        '<WEB>': 9,
        '</WEB>': 10,
        '<CODE>': 11,
        '</CODE>': 12,
        '<VALUE>': 13,
        '</VALUE>': 14,
    }
    
    def __init__(self, vocab_size=4096):
        self.vocab_size = vocab_size
        self.n_special = len(self.SPECIAL_TOKENS)
        self.char_level_vocab = {}
        self.merges = {}  # (token_a, token_b) -> new_token_id
        self.inverse_merges = {}  # new_token_id -> (token_a, token_b)
        self.vocab = {}  # token_id -> string representation
        self.trained = False
        
        # Initialize with special tokens
        for name, idx in self.SPECIAL_TOKENS.items():
            self.vocab[idx] = name
    
    def train(self, text):
        """Train BPE tokenizer on text."""
        # Pre-tokenize into words
        words = re.findall(r'\S+', text) + list(text.replace(' ', '▁'))
        
        # Get all unique characters
        chars = sorted(list(set(text)))
        
        # Build initial character vocabulary (after special tokens)
        start_idx = self.n_special
        for i, c in enumerate(chars):
            idx = start_idx + i
            self.char_level_vocab[c] = idx
            self.vocab[idx] = c
        
        current_vocab_size = start_idx + len(chars)
        
        # Convert words to token lists
        def word_to_tokens(word):
            return [self.char_level_vocab.get(c, self.SPECIAL_TOKENS['<UNK>']) for c in word]
        
        word_tokens = [word_to_tokens(w) for w in words]
        
        # Iteratively merge most frequent pairs
        while current_vocab_size < self.vocab_size:
            # Count all adjacent pairs
            pair_counts = Counter()
            for tokens in word_tokens:
                for i in range(len(tokens) - 1):
                    pair_counts[(tokens[i], tokens[i+1])] += 1
            
            if not pair_counts:
                break
            
            # Find most frequent pair
            most_common = pair_counts.most_common(1)[0]
            pair, count = most_common
            
            if count < 2:  # Stop if no pair appears more than once
                break
            
            # Create new token id for this pair
            new_id = current_vocab_size
            current_vocab_size += 1
            
            # Register the merge
            self.merges[pair] = new_id
            self.inverse_merges[new_id] = pair
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            
            # Apply merge to all words
            new_word_tokens = []
            for tokens in word_tokens:
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                        new_tokens.append(new_id)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                new_word_tokens.append(new_tokens)
            word_tokens = new_word_tokens
        
        self.trained = True
        self.vocab_size = current_vocab_size
        print(f"  BPE Vocabulary: {current_vocab_size} tokens ({len(self.merges)} merges learned)")
        return current_vocab_size
    
    def encode_single(self, word):
        """Encode a single word/token using the learned BPE merges."""
        if not self.trained:
            return [self.char_level_vocab.get(c, self.SPECIAL_TOKENS['<UNK>']) for c in word]
        
        # Start with character-level tokens
        tokens = [self.char_level_vocab.get(c, self.SPECIAL_TOKENS['<UNK>']) for c in word]
        
        # Greedily apply merges
        changed = True
        while changed:
            changed = False
            # Find the lowest-priority merge that applies first
            # (We apply merges in the order they were learned)
            for pair, new_id in sorted(self.merges.items(), key=lambda x: x[1]):
                i = 0
                new_tokens = []
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                        new_tokens.append(new_id)
                        i += 2
                        changed = True
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens
        
        return tokens
    
    def encode(self, text):
        """Encode text into token IDs."""
        tokens = [self.SPECIAL_TOKENS['<BOS>']]
        
        # Split into words and encode each
        for word in re.findall(r'\S+|\s+', text):
            tokens.extend(self.encode_single(word))
        
        tokens.append(self.SPECIAL_TOKENS['<EOS>'])
        return tokens
    
    def decode(self, token_ids):
        """Convert token IDs back to text."""
        result = []
        for tid in token_ids:
            if tid in self.vocab:
                token_str = self.vocab[tid]
                if token_str.startswith('<') and token_str.endswith('>'):
                    if token_str in ('<PAD>', '<UNK>', '<BOS>', '<EOS>', '<MASK>'):
                        continue  # Skip special tokens in output
                result.append(token_str)
            else:
                result.append('?')
        return ''.join(result)
    
    def get_vocab_size(self):
        return len(self.vocab)
    
    def save(self, path):
        import json
        data = {
            'vocab_size': self.vocab_size,
            'merges': {f"{k[0]},{k[1]}": v for k, v in self.merges.items()},
            'char_level_vocab': self.char_level_vocab,
            'vocab': {str(k): v for k, v in self.vocab.items()}
        }
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path):
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        self.vocab_size = data['vocab_size']
        self.merges = {tuple(map(int, k.split(','))): v for k, v in data['merges'].items()}
        self.inverse_merges = {v: k for k, v in self.merges.items()}
        self.char_level_vocab = {int(k) if k.isdigit() else k: v for k, v in data['char_level_vocab'].items()}
        self.vocab = {int(k): v for k, v in data['vocab'].items()}
        self.trained = True


def demo_tokenizer():
    """Demonstrate BPE tokenizer."""
    print("=" * 60)
    print("BPE TOKENIZER DEMONSTRATION")
    print("=" * 60)
    
    tokenizer = BPETokenizer(vocab_size=512)
    
    # Training text
    train_text = """
To be or not to be, that is the question.
All that glitters is not gold.
Where there is a will, there is a way.
2 + 3 = 5. Multiplication: 3 * 4 = 12. Division: 12 / 4 = 3.
def factorial(n): return 1 if n <= 1 else n * factorial(n - 1)
Knowledge should help not harm. Verify information before trusting.
    """
    
    print(f"\nTraining on {len(train_text)} chars...")
    tokenizer.train(train_text)
    
    print(f"\nVocabulary size: {tokenizer.get_vocab_size()}")
    
    # Test encoding/decoding
    test_texts = [
        "To be or not to be",
        "2 + 3 = 5",
        "def factorial(n)",
        "Ethics in AI",
        "The quick brown fox"
    ]
    
    print("\nEncoding/Decoding tests:")
    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        compression = len(text) / max(len(encoded), 1)
        print(f"  Input:    \"{text}\"")
        print(f"  Encoded:  {encoded[:20]}... ({len(encoded)} tokens)")
        print(f"  Decoded:  \"{decoded[:60]}...\"")
        print(f"  Compression: {compression:.2f}x")
        print()
    
    return tokenizer


if __name__ == "__main__":
    tokenizer = demo_tokenizer()