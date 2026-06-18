"""Rank fusion: combine multiple rankers into one ordering.

Reciprocal Rank Fusion (RRF) is the safe default — it needs no score
calibration between rankers (BM25 scores and cosine sims and rule fits live on
totally different scales), only ranks. A candidate that several rankers all
place near the top accumulates 1/(k+rank) votes and floats up; a candidate only
one ranker likes stays put.

    rrf(c) = Σ_i  w_i / (k + rank_i(c))        # k=60 is the standard constant

Every new ranker (sparse, evidence-graph, LTR, GNN) plugs in here as one more
rank list. The weighted-fit ensemble remains ranker #1 (it already bakes in the
availability multiplier, so behavioral readiness keeps its influence through the
fusion).
"""
from collections import defaultdict

RRF_K = 60  # the canonical RRF smoothing constant from the original paper


def reciprocal_rank_fusion(
    rank_lists: list[tuple[str, list[str]]],
    k: int = RRF_K,
    weights: dict[str, float] | None = None,
) -> tuple[dict[str, float], list[str]]:
    """Fuse ranked lists.

    Args:
        rank_lists: list of (name, candidate_ids_in_ranked_order). Every list
            MUST be over the same candidate universe (e.g. all post-filter).
        k: RRF smoothing constant (60 by default).
        weights: optional per-ranker multipliers (name -> weight). Default 1.0
            each, so a ranker's contribution can be down-weighted (e.g. treat
            BM25 as a weak signal) without dropping it.

    Returns:
        (scores, order) where scores maps candidate_id -> rrf score and order is
        candidate_ids sorted by (-score, candidate_id) — matching the
        validator's tie-break rule.
    """
    rrf: dict[str, float] = defaultdict(float)
    for name, ranking in rank_lists:
        w = (weights or {}).get(name, 1.0)
        for pos, cid in enumerate(ranking):
            rrf[cid] += w / (k + pos)
    order = sorted(rrf, key=lambda c: (-rrf[c], c))
    return dict(rrf), order


def rescale_to_unit(scores: dict[str, float]) -> dict[str, float]:
    """Linearly map a score dict to [0, 1] (max -> 1, min -> 0).

    Used when we want to blend a ranker's raw scores into the weighted-fit
    ensemble as a normalised component rather than via RRF."""
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi <= lo:
        return {c: 1.0 for c in scores}
    return {c: (v - lo) / (hi - lo) for c, v in scores.items()}
