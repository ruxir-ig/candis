"""Sparse lexical ranker: a from-scratch BM25 over candidate text.

Why hand-rolled instead of `rank-bm25`: BM25 is ~40 lines, and keeping the
runtime dependency surface to numpy-only matters for the challenge's 5-min /
no-network ranking budget. This module is the *offline* precompute scorer; the
runtime path only loads the cached ranks it produces (see fusion.py).

This is the sparse complement to the dense bi-encoder (semantic_matcher): both
score candidates against the same distilled ideal-candidate query, but BM25
rewards *exact* technical-term overlap (FAISS, Qdrant, NDCG, retrieval...) while
the embedding rewards paraphrase. Textbook hybrid retrieval.
"""
import math
import re
from collections import Counter, defaultdict

_TOKEN_RE = re.compile(r"[a-z0-9+#]+")

# Compact stopword list — kept small on purpose. Technical terms like "search"
# or "rank" are NOT here (they carry signal); only grammatical glue is dropped.
_STOPWORDS = frozenset(
    "a an and the of to in for with on at by from is are be as it its this that "
    "we our you your i my me have has had do does did will would can could should "
    "not no or if then so than into over per via etc and/or them they their he she "
    "his her who whom whose which what when where why how all any each few more most "
    "other some such only own same too very just also but while during about above "
    "below up down out off again further once here there across after before between "
    "through among".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, pull alphanumeric+#+# tokens (so C++, C#, etc. survive), drop
    stopwords and 1-char tokens."""
    if not text:
        return []
    return [
        t for t in _TOKEN_RE.findall(text.lower())
        if len(t) >= 2 and t not in _STOPWORDS
    ]


class BM25Index:
    """Okapi BM25 over a fixed corpus. Build once offline, score many queries."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.doc_len: list[int] = []
        self.tf: list[Counter] = []          # per-doc term frequencies
        self.df: dict[str, int] = defaultdict(int)  # document frequency per term
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)  # term -> [(doc_idx, tf)]
        self.avgdl: float = 0.0
        self.idf: dict[str, float] = {}

    def build(self, docs: list[tuple[str, str]]):
        """docs = list of (candidate_id, text)."""
        n = len(docs)
        total_len = 0
        for cid, text in docs:
            toks = tokenize(text)
            counts = Counter(toks)
            self.doc_ids.append(cid)
            self.doc_len.append(len(toks))
            self.tf.append(counts)
            total_len += len(toks)
            i = len(self.tf) - 1
            for term, f in counts.items():
                self.df[term] += 1
                self.postings[term].append((i, f))
        self.avgdl = total_len / n if n else 0.0
        # Okapi idf with the +1 smoothing so no term is ever negative.
        for term, df in self.df.items():
            self.idf[term] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        return self

    def score(self, query_tokens: list[str]) -> dict[str, float]:
        """Return {candidate_id: bm25_score} for the whole corpus.

        Uses the inverted postings index so we only touch docs that actually
        contain a query term (O(sum of postings), not O(terms x corpus))."""
        scores = {cid: 0.0 for cid in self.doc_ids}
        k1, b, avgdl = self.k1, self.b, self.avgdl
        for term in set(query_tokens):
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for i, f in self.postings[term]:
                dl = self.doc_len[i]
                denom = f + k1 * (1 - b + b * dl / avgdl)
                scores[self.doc_ids[i]] += idf * f * (k1 + 1) / denom
        return scores
