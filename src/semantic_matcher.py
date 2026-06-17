"""Hot-path semantic scoring — numpy only, no torch.

Loads the precomputed embeddings and turns each candidate's cosine similarity to
the JD query into a [0,1] score. Cosine sims are robustly min-max normalized
(1st/99th percentile) so the component sits on the same scale as the rule-based
scorers in the ensemble.

If the cache is missing, semantic_scores() returns None and the ranker falls
back to rule-only weights — so rank.py never hard-depends on the precompute.
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache/embeddings.npz"


def semantic_scores(cache_path: Path = CACHE):
    """Return {candidate_id: semantic_score in [0,1]} or None if no cache."""
    if not Path(cache_path).exists():
        return None

    data = np.load(cache_path, allow_pickle=True)
    ids = data["candidate_ids"]
    emb = data["embeddings"]            # N x D, L2-normalized
    jd = data["jd_embedding"]           # D, L2-normalized

    sims = emb @ jd                     # cosine similarity (both normalized)

    lo, hi = np.percentile(sims, 1), np.percentile(sims, 99)
    scaled = np.clip((sims - lo) / (hi - lo + 1e-9), 0.0, 1.0)

    return {str(cid): float(s) for cid, s in zip(ids, scaled)}
