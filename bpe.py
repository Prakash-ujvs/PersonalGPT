"""Minimal pure-Python BPE tokenizer (no torch needed)."""
import json, re
from collections import Counter

BASE = " .abcdefghijklmnopqrstuvwxyz"
_idx = {c: i for i, c in enumerate(BASE)}

_pre = re.compile(r" ?[a-z]+|\.| ")

def clean(t): return re.sub(r"[^a-z .]", "", re.sub(r"\s+", " ", t.lower()))
def pretokens(t): return _pre.findall(clean(t))

class BPE:
    def __init__(self, merges=()):
        self.merges = [tuple(m) for m in merges]
        self.vocab = list(BASE)
        for a, b in self.merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self.size = len(self.vocab)
        self._cache = {}

    @staticmethod
    def _merge(ids, pair, new):
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                out.append(new); i += 2
            else:
                out.append(ids[i]); i += 1
        return out

    @classmethod
    def train(cls, text, vocab_size=256):
        words = Counter(pretokens(text))                       # merges never cross word boundaries
        seqs = {w: [_idx[c] for c in w] for w in words}
        merges = []
        while len(BASE) + len(merges) < vocab_size:
            pairs = Counter()
            for w, ids in seqs.items():
                for p in zip(ids, ids[1:]): pairs[p] += words[w]
            if not pairs: break
            pair, n = pairs.most_common(1)[0]
            if n < 2: break
            new = len(BASE) + len(merges)
            for w in seqs: seqs[w] = cls._merge(seqs[w], pair, new)
            merges.append(pair)
        return cls(merges)

    def _word(self, w):
        r = self._cache.get(w)
        if r is None:
            r = [_idx[c] for c in w]
            for i, pair in enumerate(self.merges):
                r = self._merge(r, pair, len(BASE) + i)
            self._cache[w] = r
        return r

    def encode(self, text):
        out = []
        for w in pretokens(text): out += self._word(w)
        return out

    def decode(self, ids): return "".join(self.vocab[i] for i in ids)

    def save(self, path="tokenizer.json"):
        json.dump(self.merges, open(path, "w"))

    @classmethod
    def load(cls, path="tokenizer.json"): return cls(json.load(open(path)))