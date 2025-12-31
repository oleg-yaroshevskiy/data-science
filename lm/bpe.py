import json
import re
from collections import defaultdict

BPE_VOCAB_SIZE = 30000

# Pre-compile regex pattern for better performance
WORD_PATTERN = re.compile(r'\s*\S+')

# Special tokens
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
BOS_TOKEN = '<BOS>'
EOS_TOKEN = '<EOS>'

class SimpleBPE:
    def __init__(self, vocab_file: str = None, vocab_size: int = BPE_VOCAB_SIZE):
        """
        Initialize SimpleBPE with a vocabulary file and size.
        """
        self.vocab_file = vocab_file
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        self.merges = []  # Store merge order: list of (token1, token2) pairs
        
        if self.vocab_file:
            self.load_vocab(self.vocab_file)
        else:
            # Initialize with special tokens
            self._init_special_tokens()
    
    def load_vocab(self, vocab_file: str):
        """Load vocabulary and merges from file."""
        try:
            with open(vocab_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    # Old format: just token_to_id
                    self.token_to_id = data
                    self.merges = []
                else:
                    # New format: {"token_to_id": {...}, "merges": [...]}
                    self.token_to_id = data.get('token_to_id', data)
                    self.merges = [tuple(m) for m in data.get('merges', [])]
                self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        except Exception as e:
            print(f"Error loading vocabulary from {vocab_file}: {e}")
            raise e
    
    def _init_special_tokens(self):
        """Initialize vocabulary with special tokens."""
        special_tokens = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
        for idx, token in enumerate(special_tokens):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
    
    def fit(self, texts: list[str], verbose: bool = True):
        """
        Fit SimpleBPE on the provided texts to build vocabulary.
        
        Args:
            texts: List of text strings to train on
            verbose: Whether to print progress during training
        """
        # Initialize with special tokens
        self._init_special_tokens()
        
        # Add all unique characters from texts
        _vocab = set((char for text in texts for char in text))
        for char in _vocab:
            if char not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[char] = idx
                self.id_to_token[idx] = char

        corpora = [self.tokenize(text) for text in texts]
        
        while len(self.token_to_id) < self.vocab_size:
            pairs = self.get_stats(corpora)
            if not pairs:
                break
            new_best_pair = max(pairs, key=pairs.get)

            new_token = self.id_to_token[new_best_pair[0]] + self.id_to_token[new_best_pair[1]]

            print("Adding new token:", new_token.replace(" ", "_"))

            self.token_to_id[new_token] = len(self.token_to_id)
            self.id_to_token[len(self.id_to_token)] = new_token
            
            # Store the merge in order
            self.merges.append((self.id_to_token[new_best_pair[0]], self.id_to_token[new_best_pair[1]]))

            corpora = self.update_corpora(new_best_pair, corpora)
    
    def get_stats(self, corpora: list[list]) -> dict[tuple[int, int], int]:
        """Count frequency of adjacent token pairs in corpora."""
        pairs = defaultdict(int)
        for sequence in corpora:
            for i in range(len(sequence) - 1):
                pair = (sequence[i], sequence[i + 1])
                pairs[pair] += 1
        return dict(pairs)

    def tokenize(self, text: str) -> list[int]:
        """Tokenize text by applying BPE merges in learned order."""
        # Split text into words (with leading whitespace preserved)
        words = WORD_PATTERN.findall(text)
        
        tokens = []
        for word in words:
            # Start with character-level tokens
            word_tokens = list(word)
            
            # Apply merges in the order they were learned
            for merge_pair in self.merges:
                i = 0
                new_word_tokens = []
                while i < len(word_tokens):
                    # Check if current and next token match the merge pair
                    if (i < len(word_tokens) - 1 and 
                        word_tokens[i] == merge_pair[0] and 
                        word_tokens[i + 1] == merge_pair[1]):
                        # Merge them
                        new_word_tokens.append(merge_pair[0] + merge_pair[1])
                        i += 2
                    else:
                        new_word_tokens.append(word_tokens[i])
                        i += 1
                word_tokens = new_word_tokens
            
            # Convert tokens to IDs
            unk_id = self.token_to_id.get(UNK_TOKEN, 1)
            for token in word_tokens:
                tokens.append(self.token_to_id.get(token, unk_id))
        
        return tokens
    
    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text."""
        tokens = [self.id_to_token.get(token_id, UNK_TOKEN) for token_id in token_ids]
        # Join tokens and handle special tokens
        text = ''.join(tokens)
        # Remove special tokens if present
        for special in [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]:
            text = text.replace(special, '')
        return text
    
    def save_vocab(self, vocab_file: str):
        """Save vocabulary and merges to file."""
        data = {
            'token_to_id': self.token_to_id,
            'merges': self.merges
        }
        with open(vocab_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def update_corpora(self, pair: tuple[int, int], corpora: list[list[int]]) -> list[list[int]]:
        """Update corpora by replacing pair occurrences with merged token."""
        new_token_id = self.token_to_id[self.id_to_token[pair[0]] + self.id_to_token[pair[1]]]
        new_corpora = []
        for sequence in corpora:
            new_sequence = []
            i = 0
            while i < len(sequence):
                if i < len(sequence) - 1 and (sequence[i], sequence[i + 1]) == pair:
                    new_sequence.append(new_token_id)
                    i += 2
                else:
                    new_sequence.append(sequence[i])
                    i += 1
            new_corpora.append(new_sequence)
        return new_corpora
    

if __name__ == "__main__":
    input_file = "../wikitext103_train.txt"
    with open(input_file, 'r') as f:
        texts = f.readlines()[:10000]
    
    bpe = SimpleBPE(vocab_size=1000)
    bpe.fit(texts)

    # Save vocabulary
    bpe.save_vocab("bpe_vocab.json")
    
    # Test tokenization and decoding
    test_text = "Hello world!"
    token_ids = bpe.tokenize(test_text)
    decoded = bpe.decode(token_ids)
    print(f"Original: {test_text}")
    print(f"Token IDs: {token_ids}")
    print(f"Decoded: {decoded}")