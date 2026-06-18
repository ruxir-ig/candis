#!/usr/bin/env python3
"""Offline precompute: BM25 sparse scores for the full candidate pool.

Builds an Okapi BM25 index over candidate text (the same distilled
ideal-candidate query the dense encoder uses), scores every post-filter
candidate, and writes a normalised score cache the runtime ranker loads.

Anti-gaming gate (the keyword-stuffer fix from the plan): a candidate whose BM25
is high but whose *structured* title and career scores are low almost certainly
pasted AI keywords into prose that doesn't back them up. We damp those hard so
BM25 can only help, never carry, a fake.

    uv run python precompute/build_sparse.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.loader import load_candidates  # noqa: E402
from src.honeypot_detector import is_honeypot  # noqa: E402
from src.coarse_filter import passes_coarse_filter  # noqa: E402
from src.profile_text import candidate_text  # noqa: E402
from src.jd_query import IDEAL_CANDIDATE_QUERY  # noqa: E402
from src.sparse_matcher import BM25Index, tokenize  # noqa: E402
from src.scorers import title_role_score, career_score  # noqa: E402

CACHE = ROOT / "cache" / "sparse_scores.npz"

# Gate thresholds (plan's anti-stuffing fix).
GATE_SPARSE = 0.75      # normalised BM25 considered "high"
GATE_TITLE = 0.40       # structured title score below this = no corroboration
GATE_CAREER = 0.40      # structured career score below this = no corroboration
GATE_PENALTY = 0.15     # multiply BM25 by this when the gate trips


def build_doc(candidate: dict) -> str:
    """Document for BM25: the embed prose plus explicit career titles so exact
    role terms (Search Engineer, Staff ML Engineer) get lexical weight too."""
    parts = [candidate_text(candidate)]
    titles = [r.get("title", "") for r in candidate.get("career_history", [])[:6]]
    p = candidate.get("profile", {})
    if p.get("current_title"):
        titles.append(p["current_title"])
    parts.append(" ".join(t for t in titles if t))
    return " \n".join(parts)


def main():
    t0 = time.time()
    candidates = list(load_candidates(ROOT / "data" / "candidates.jsonl"))
    print(f"Loaded {len(candidates):,} candidates in {time.time()-t0:.1f}s")

    # Same hard filter as the ranker — BM25 ranks only the recall set.
    kept = []
    for c in candidates:
        if is_honeypot(c):
            continue
        if not passes_coarse_filter(c):
            continue
        kept.append(c)
    print(f"Post-filter: {len(kept):,} candidates scored by BM25")

    docs = [(c["candidate_id"], build_doc(c)) for c in kept]

    t1 = time.time()
    idx = BM25Index().build(docs)
    print(f"Built BM25 index ({len(idx.df):,} unique terms) in {time.time()-t1:.1f}s")

    q_tokens = tokenize(IDEAL_CANDIDATE_QUERY)
    scores = idx.score(q_tokens)

    # Normalise raw BM25 to [0, 1] (max candidate -> 1).
    raw = np.array([scores[cid] for cid in idx.doc_ids], dtype=np.float64)
    hi = raw.max()
    norm = raw / hi if hi > 0 else raw
    norm_map = {cid: float(norm[i]) for i, cid in enumerate(idx.doc_ids)}

    # Contradiction gate: damp BM25 where it isn't corroborated structurally.
    gated = 0
    for c in kept:
        cid = c["candidate_id"]
        if norm_map[cid] > GATE_SPARSE and title_role_score(c) < GATE_TITLE \
                and career_score(c) < GATE_CAREER:
            norm_map[cid] *= GATE_PENALTY
            gated += 1

    cids = [c["candidate_id"] for c in kept]
    vals = np.array([norm_map[c] for c in cids], dtype=np.float32)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, candidate_ids=np.array(cids), scores=vals)
    print(f"Gated {gated:,} keyword-stuffer suspects (BM25 high, title+career low)")
    print(f"Wrote {CACHE.relative_to(ROOT)} — top BM25: "
          f"{vals.max():.3f}, mean {vals.mean():.3f}, nonzero {(vals>0).sum():,}")
    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
