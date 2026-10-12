"""Minimal pure-Python BPE tokenizer (no torch needed)."""
import json
import re
from collections import Counter
from threading import Lock

BASE = " .abcdefghijklmnopqrstuvwxyz"
_idx = {c: i for i, c in enumerate(BASE)}

_pre = re.compile(r" ?[a-z]+|\.| ")


def clean(t: str) -> str:
    return re.sub(r"[^a-z .]", "", re.sub(r"\s+", " ", t.lower()))


def pretokens(t: str) -> list[str]:
    return _pre.findall(clean(t))


class BPE:
    def __init__(self, merges=()):
        self.merges = [tuple(m) for m in merges]
        self.vocab = list(BASE)
        for a, b in self.merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self.size = len(self.vocab)
        self._cache = {}
        self._lock = Lock()

    @staticmethod
    def _merge(ids: list[int], pair: tuple[int, int], new: int) -> list[int]:
        if pair[0] not in ids or pair[1] not in ids:
            return ids
        out, i = [], 0
        n = len(ids)
        while i < n:
            if i < n - 1 and (ids[i], ids[i + 1]) == pair:
                out.append(new)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    @classmethod
    def train(cls, text: str, vocab_size: int = 256):
        words = Counter(pretokens(text))  # merges never cross word boundaries
        seqs = {w: [_idx[c] for c in w] for w in words}
        merges = []
        while len(BASE) + len(merges) < vocab_size:
            pairs = Counter()
            for w, ids in seqs.items():
                w_count = words[w]
                for p in zip(ids, ids[1:]):
                    pairs[p] += w_count
            if not pairs:
                break
            pair, n = pairs.most_common(1)[0]
            if n < 2:
                break
            new = len(BASE) + len(merges)
            for w, ids in seqs.items():
                if pair[0] in ids and pair[1] in ids:
                    seqs[w] = cls._merge(ids, pair, new)
            merges.append(pair)
        return cls(merges)

    def _word(self, w: str) -> list[int]:
        with self._lock:
            r = self._cache.get(w)
        if r is not None:
            return r

        r = [_idx[c] for c in w]
        for i, pair in enumerate(self.merges):
            if pair[0] in r and pair[1] in r:
                r = self._merge(r, pair, len(BASE) + i)

        with self._lock:
            self._cache[w] = r
        return r

    def encode(self, text: str) -> list[int]:
        out = []
        for w in pretokens(text):
            out.extend(self._word(w))
        return out

    def decode(self, ids: list[int]) -> str:
        return "".join(self.vocab[i] for i in ids if 0 <= i < self.size)

    def save(self, path: str = "tokenizer.json") -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.merges, f, indent=2)

    @classmethod
    def load(cls, path: str = "tokenizer.json"):
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))