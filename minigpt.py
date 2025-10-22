# tokenizers.py
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import re
from typing import Dict, List, Tuple, Iterable, Optional

# ---- base ---------------------------------------------------------
SPECIAL_TOKENS = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<unk>": 3,
}

@dataclass
class TokenizerConfig:
    vocab_size: int = 30000
    lowercase: bool = False
    byte_level: bool = False   # used by BPE variant

class BaseTokenizer:
    name: str = "base"

    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()
        self.stoi: Dict[str, int] = dict(SPECIAL_TOKENS)  # string->id
        self.itos: List[str] = [None] * len(SPECIAL_TOKENS)
        for k, v in SPECIAL_TOKENS.items():
            self.itos[v] = k

    # --- API you will use ----------------------------------------------------
    def train(self, texts: Iterable[str]) -> None:
        raise NotImplementedError

    def encode(self, text: str, add_bos_eos: bool = True) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: List[int], skip_special=True) -> str:
        toks = [self.itos[i] if 0 <= i < len(self.itos) else "<unk>" for i in ids]
        if skip_special:
            toks = [t for t in toks if t not in SPECIAL_TOKENS]
        return self._detokenize(toks)

    # --- helpers -------------------------------------------------------------
    def _tokenize(self, text: str) -> List[str]:
        return [text]  # override

    def _detokenize(self, toks: List[str]) -> str:
        return "".join(toks)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    # --- persistence ---------------------------------------------------------
    def save(self, path: str) -> None:
        meta = {
            "name": self.name,
            "config": self.config.__dict__,
            "stoi": self.stoi,
            "itos": self.itos,
            "extra": getattr(self, "_extra", None),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

    @staticmethod
    def load(path: str) -> "BaseTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        name = meta["name"]
        tok = TOKENIZER_REGISTRY[name](TokenizerConfig(**meta["config"]))
        tok.stoi = {k: int(v) if isinstance(v, str) and v.isdigit() else v for k, v in meta["stoi"].items()}
        tok.itos = meta["itos"]
        tok._extra = meta.get("extra", None)
        return tok

# ---- char-level ----------------------------------------------------
class CharTokenizer(BaseTokenizer):
    name = "char"

    def _tokenize(self, text: str) -> List[str]:
        if self.config.lowercase:
            text = text.lower()
        return list(text)

    def train(self, texts: Iterable[str]) -> None:
        freq = Counter()
        for t in texts:
            freq.update(self._tokenize(t))
        # reserve slots for specials
        most_common = [c for c, _ in freq.most_common(self.config.vocab_size - len(SPECIAL_TOKENS))]
        for idx, token in enumerate(most_common, start=len(SPECIAL_TOKENS)):
            self.stoi[token] = idx
            if len(self.itos) <= idx:
                self.itos.append(token)
            else:
                self.itos[idx] = token

    def encode(self, text: str, add_bos_eos: bool = True) -> List[int]:
        toks = self._tokenize(text)
        ids = [self.stoi.get(t, SPECIAL_TOKENS["<unk>"]) for t in toks]
        if add_bos_eos:
            ids = [SPECIAL_TOKENS["<bos>"]] + ids + [SPECIAL_TOKENS["<eos>"]]
        return ids

# ---- simple whitespace word-level ---------------------------------
class WordTokenizer(BaseTokenizer):
    name = "word"

    def _tokenize(self, text: str) -> List[str]:
        if self.config.lowercase:
            text = text.lower()
        # simple word split; keep punctuation as separate tokens
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

    def _detokenize(self, toks: List[str]) -> str:
        # naive join with spaces between alphanumerics
        out = []
        for i, t in enumerate(toks):
            if i > 0 and re.match(r"\w", t) and re.match(r"\w", toks[i-1]):
                out.append(" ")
            out.append(t)
        return "".join(out)

    def train(self, texts: Iterable[str]) -> None:
        freq = Counter()
        for t in texts:
            freq.update(self._tokenize(t))
        vocab = [w for w, _ in freq.most_common(self.config.vocab_size - len(SPECIAL_TOKENS))]
        for idx, token in enumerate(vocab, start=len(SPECIAL_TOKENS)):
            self.stoi[token] = idx
            if len(self.itos) <= idx:
                self.itos.append(token)
            else:
                self.itos[idx] = token

    def encode(self, text: str, add_bos_eos: bool = True) -> List[int]:
        toks = self._tokenize(text)
        ids = [self.stoi.get(t, SPECIAL_TOKENS["<unk>"]) for t in toks]
        if add_bos_eos:
            ids = [SPECIAL_TOKENS["<bos>"]] + ids + [SPECIAL_TOKENS["<eos>"]]
        return ids

# ---- minimal BPE (character-level merges) --------------------------
# NOTE: This is a compact educational BPE (merges frequent adjacent pairs of symbols).
# It’s not byte-level GPT-2 BPE, but works well as a demo.
class SimpleBPE(BaseTokenizer):
    name = "bpe"

    def train(self, texts: Iterable[str]) -> None:
        # Start from character tokens (including spaces as tokens)
        def split_chars(s: str) -> List[str]:
            return list(s) if not self.config.lowercase else list(s.lower())

        # build initial corpus as lists of symbols
        corpus: List[List[str]] = [split_chars(t) + ["</w>"] for t in texts]  # </w> end-of-line marker
        vocab = defaultdict(int)
        for word in corpus:
            vocab[tuple(word)] += 1

        merges: List[Tuple[str, str]] = []
        target_merges = max(0, self.config.vocab_size - len(SPECIAL_TOKENS) - 500)  # heuristic cap

        def get_pair_stats(vocab) -> Counter:
            pairs = Counter()
            for word, count in vocab.items():
                for i in range(len(word) - 1):
                    pairs[(word[i], word[i+1])] += count
            return pairs

        def merge_vocab(pair, vocab):
            bigram = pair
            new_vocab = defaultdict(int)
            first, second = bigram
            for word, count in vocab.items():
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == first and word[i+1] == second:
                        new_word.append(first + second)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_vocab[tuple(new_word)] += count
            return new_vocab

        for _ in range(target_merges):
            pairs = get_pair_stats(vocab)
            if not pairs:
                break
            best = pairs.most_common(1)[0][0]
            vocab = merge_vocab(best, vocab)
            merges.append(best)

        # Build final symbol vocab from vocab entries
        sym_freq = Counter()
        for word, count in vocab.items():
            sym_freq.update({sym: count for sym in set(word)})
        symbols = [s for s, _ in sym_freq.most_common(self.config.vocab_size - len(SPECIAL_TOKENS))]
        for idx, token in enumerate(symbols, start=len(SPECIAL_TOKENS)):
            self.stoi[token] = idx
            if len(self.itos) <= idx:
                self.itos.append(token)
            else:
                self.itos[idx] = token

        self._extra = {"merges": merges}

    def _apply_merges(self, tokens: List[str]) -> List[str]:
        merges: List[Tuple[str, str]] = self._extra.get("merges", [])
        if not merges:
            return tokens
        token_list = tokens[:]
        for a, b in merges:
            i = 0
            out = []
            while i < len(token_list):
                if i < len(token_list) - 1 and token_list[i] == a and token_list[i+1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(token_list[i])
                    i += 1
            token_list = out
        return token_list

    def encode(self, text: str, add_bos_eos: bool = True) -> List[int]:
        s = text.lower() if self.config.lowercase else text
        chars = list(s) + ["</w>"]
        toks = self._apply_merges(chars)
        ids = [self.stoi.get(t, SPECIAL_TOKENS["<unk>"]) for t in toks]
        if add_bos_eos:
            ids = [SPECIAL_TOKENS["<bos>"]] + ids + [SPECIAL_TOKENS["<eos>"]]
        return ids

    def _detokenize(self, toks: List[str]) -> str:
        text = "".join(toks)
        return text.replace("</w>", "")

# ---- registry ------------------------------------------------------
TOKENIZER_REGISTRY = {
    CharTokenizer.name: CharTokenizer,
    WordTokenizer.name: WordTokenizer,
    SimpleBPE.name: SimpleBPE,
}

def build_tokenizer(name: str, config: Optional[TokenizerConfig] = None) -> BaseTokenizer:
    name = name.lower()
    if name not in TOKENIZER_REGISTRY:
        raise ValueError(f"Unknown tokenizer '{name}'. Options: {list(TOKENIZER_REGISTRY)}")
    return TOKENIZER_REGISTRY[name](config or TokenizerConfig())
    
    
# train.py
import argparse
from pathlib import Path
import torch
from tokenizers import build_tokenizer, TokenizerConfig, BaseTokenizer, TOKENIZER_REGISTRY, SPECIAL_TOKENS

def read_corpus(path: str):
    txt = Path(path).read_text(encoding="utf-8")
    # simple split into "lines"/samples for tokenizer training
    return [ln for ln in txt.splitlines() if ln.strip()]

def choose_tokenizer_interactively() -> str:
    print("Choose a tokenizer:")
    for i, name in enumerate(TOKENIZER_REGISTRY.keys(), 1):
        print(f"  {i}. {name}")
    idx = input("Enter number (default=1): ").strip() or "1"
    keys = list(TOKENIZER_REGISTRY.keys())
    i = max(1, min(int(idx), len(keys))) - 1
    return keys[i]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="Path to training text file")
    p.add_argument("--tokenizer", type=str, default=None, help="char|word|bpe (optional; prompts if omitted)")
    p.add_argument("--vocab-size", type=int, default=30000)
    p.add_argument("--lowercase", action="store_true")
    p.add_argument("--tok-file", type=str, default=None, help="Optional path to save/load tokenizer JSON")
    p.add_argument("--seq-len", type=int, default=128)
    args = p.parse_args()

    tok_name = args.tokenizer or choose_tokenizer_interactively()
    cfg = TokenizerConfig(vocab_size=args.vocab_size, lowercase=args.lowercase)
    tok: BaseTokenizer = build_tokenizer(tok_name, cfg)

    texts = read_corpus(args.data)

    # load if exists, else train + save (if path provided)
    if args.tok_file and Path(args.tok_file).exists():
        print(f"Loading tokenizer from {args.tok_file}")
        tok = BaseTokenizer.load(args.tok_file)
    else:
        print(f"Training {tok_name} tokenizer on {len(texts)} lines …")
        tok.train(texts)
        if args.tok_file:
            tok.save(args.tok_file)
            print(f"Saved tokenizer to {args.tok_file}")

    sample = texts[0]
    ids = tok.encode(sample)
    print(f"\nSample text: {sample[:80]!r}")
    print(f"Encoded ids (len={len(ids)}): {ids[:40]}{' …' if len(ids)>40 else ''}")
    print(f"Decoded back: {tok.decode(ids)}")

    # ---- make a toy batch tensor for your model training -------------------
    # concat all text and turn into token ids
    all_ids = []
    for t in texts:
        all_ids.extend(tok.encode(t))
    data = torch.tensor(all_ids, dtype=torch.long)

    # make simple sequential batches (for bigram/transformer later)
    def get_batch(bs=32, seq_len=args.seq_len):
        ix = torch.randint(0, len(data) - seq_len - 1, (bs,))
        x = torch.stack([data[i:i+seq_len] for i in ix])
        y = torch.stack([data[i+1:i+1+seq_len] for i in ix])
        return x, y  # ready for training

    x, y = get_batch()
    print(f"\nBatch shapes → x:{tuple(x.shape)} y:{tuple(y.shape)}  (vocab={tok.vocab_size})")
    print("Torch version:", torch.__version__)
    print("Ready to plug into your Bigram/Transformer model.")

if __name__ == "__main__":
    main()
    
# Example: interactive choice
python train.py --data data/tiny_shakespeare.txt --tok-file artifacts/tok_char.json

# Or pick explicitly
python train.py --data data/tiny_shakespeare.txt --tokenizer char --vocab-size 1000 --tok-file artifacts/tok_char.json
python train.py --data data/tiny_shakespeare.txt --tokenizer word --vocab-size 20000 --tok-file artifacts/tok_word.json --lowercase
python train.py --data data/tiny_shakespeare.txt --tokenizer bpe --vocab-size 8000 --tok-file artifacts/tok_bpe.json

# prep_bible.py
from pathlib import Path
import re

raw = Path("data/kjv_raw.txt").read_text(encoding="utf-8")

# remove Gutenberg headers/footers (rough but works for KJV HTML->txt dumps)
start = re.search(r"^Genesis 1:1", raw, flags=re.M)
end   = re.search(r"^Revelation 22:21", raw, flags=re.M)
text  = raw[start.start(): end.end()] if start and end else raw

# optional: normalize whitespace
text = re.sub(r"[ \t]+", " ", text)
text = re.sub(r"\n{3,}", "\n\n", text).strip()

Path("data/bible_corpus.txt").write_text(text, encoding="utf-8")
print("Saved:", "data/bible_corpus.txt", "chars:", len(text))
