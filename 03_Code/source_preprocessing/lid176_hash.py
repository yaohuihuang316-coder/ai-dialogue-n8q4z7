"""fastText feature hashing, vendored from TigreGotico/glotlid-onnx's
``glotlid_hash.py`` (https://huggingface.co/TigreGotico/glotlid-onnx).

Kept logically identical on purpose, including the sign-extension of bytes
>= 0x80 before the FNV-1a XOR: fastText's C++ ``Dictionary::hash`` treats
each byte as a signed ``int8_t`` before XOR-ing it into the hash, so
non-ASCII n-grams must be hashed the same way here or every non-Latin-script
bucket lookup breaks.
"""

from __future__ import annotations

import json
from typing import Dict, List, Sequence

import numpy as np

EOS = "</s>"
BOW = "<"
EOW = ">"
SEPARATORS = " \t\n\v\f\r"


def fnv1a(s: str) -> int:
    h = 2166136261
    for b in s.encode("utf-8"):
        if b >= 0x80:
            b |= 0xFFFFFF00
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return h


def tokenize(text: str) -> List[str]:
    for sep in SEPARATORS[1:]:
        text = text.replace(sep, " ")
    return [t for t in text.split(" ") if t] + [EOS]


def compute_subwords(word: str, minn: int, maxn: int, bucket: int,
                      nwords: int) -> List[int]:
    chars = list(word)
    out: List[int] = []
    n = len(chars)
    for i in range(n):
        for j in range(i + minn, min(n, i + maxn) + 1):
            ngram = "".join(chars[i:j])
            out.append(nwords + fnv1a(ngram) % bucket)
    return out


class FastTextFeaturizer:
    """Maps raw text to the fastText feature ids the ONNX graph expects."""

    def __init__(self, words: Sequence[str], nwords: int, minn: int = 2,
                 maxn: int = 5, bucket: int = 1_000_000):
        self.words: List[str] = list(words)
        self.nwords = nwords
        self.minn = minn
        self.maxn = maxn
        self.bucket = bucket
        self.word2id: Dict[str, int] = {w: i for i, w in enumerate(self.words)}
        self._cache: Dict[int, List[int]] = {}

    @classmethod
    def from_files(cls, vocab_path: str, meta_path: str) -> "FastTextFeaturizer":
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        with open(vocab_path, encoding="utf-8") as fh:
            words = fh.read().split("\n")
        if words and words[-1] == "":
            words.pop()
        return cls(words, meta["nwords"], meta["minn"], meta["maxn"],
                    meta["bucket"])

    def subwords_of_known(self, wid: int) -> List[int]:
        cached = self._cache.get(wid)
        if cached is None:
            word = self.words[wid]
            if word == EOS:
                cached = [wid]
            else:
                cached = [wid] + compute_subwords(
                    BOW + word + EOW, self.minn, self.maxn, self.bucket,
                    self.nwords)
            self._cache[wid] = cached
        return cached

    def add_subwords(self, line: List[int], token: str) -> None:
        wid = self.word2id.get(token, -1)
        if wid < 0:
            if token != EOS:
                line.extend(compute_subwords(BOW + token + EOW, self.minn,
                                              self.maxn, self.bucket,
                                              self.nwords))
        elif self.maxn <= 0:
            line.append(wid)
        else:
            line.extend(self.subwords_of_known(wid))

    def __call__(self, text: str) -> np.ndarray:
        line: List[int] = []
        for token in tokenize(text):
            self.add_subwords(line, token)
        return np.asarray(line, dtype=np.int64)


class HSCombiner:
    """Combines the ONNX graph's per-Huffman-node ``node_probs`` into
    per-label probabilities for hierarchical-softmax fastText models
    (e.g. fastText's own ``lid.176.bin``).

    fastText's hierarchical softmax does not reduce to a flat softmax over
    the output matrix: each row of the output matrix is a binary classifier
    for one internal node of a Huffman tree built over the label
    frequencies, and a label's probability is the product of the sigmoid
    (or 1-sigmoid) values along the root-to-leaf path. Tree construction
    (``hs_tree.build_tree``) is deterministic given the label counts, so it
    is precomputed once at export time into ``hs_tree.json``; this class
    just walks the paths, which is cheap pure-Python work analogous to the
    string hashing above - there is no portable ONNX op for it.
    """

    def __init__(self, paths: List[List[int]], codes: List[List[bool]]):
        self.paths = paths
        self.codes = codes

    @classmethod
    def from_file(cls, path: str) -> "HSCombiner":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(data["paths"], data["codes"])

    def __call__(self, node_probs: np.ndarray) -> np.ndarray:
        """node_probs: sigmoid(dot(hidden, node)) for every Huffman node.
        Returns: probability per label, same order as labels.json."""
        log_f = np.log(np.clip(node_probs, 1e-12, 1.0))
        log_1mf = np.log(np.clip(1.0 - node_probs, 1e-12, 1.0))
        out = np.empty(len(self.paths), dtype=np.float64)
        for i, (path, code) in enumerate(zip(self.paths, self.codes)):
            s = 0.0
            for node, bit in zip(path, code):
                s += log_f[node] if bit else log_1mf[node]
            out[i] = s
        return np.exp(out)
