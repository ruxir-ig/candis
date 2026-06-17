"""Ranking metrics — the same family the challenge scores on (NDCG/MAP/P@k).

Implementing these ourselves is the point: it's how you learn what the score
actually rewards. NDCG@10 is 50% of the official composite, so getting the top
of the list right matters far more than the tail.

All functions take `rels`: the graded relevances (0-4) of candidates in the
order your ranker placed them.
"""
import math

RELEVANT_THRESHOLD = 3  # grade >= 3 counts as "relevant" (per submission spec)


def dcg(rels) -> float:
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(rels, k: int) -> float:
    """rels is in ranked order; ideal is the best achievable from the same labels."""
    ideal = sorted(rels, reverse=True)
    denom = dcg(ideal[:k])
    if denom == 0:
        return 0.0
    return dcg(rels[:k]) / denom


def precision_at_k(rels, k: int, threshold: int = RELEVANT_THRESHOLD) -> float:
    top = rels[:k]
    if not top:
        return 0.0
    return sum(1 for r in top if r >= threshold) / len(top)


def average_precision(rels, threshold: int = RELEVANT_THRESHOLD) -> float:
    hits, score = 0, 0.0
    total_pos = sum(1 for r in rels if r >= threshold)
    if total_pos == 0:
        return 0.0
    for i, r in enumerate(rels, start=1):
        if r >= threshold:
            hits += 1
            score += hits / i
    return score / total_pos


def composite(rels) -> dict:
    """The official weighting, computed over whatever labeled set we have."""
    m = {
        "ndcg@10": ndcg_at_k(rels, 10),
        "ndcg@50": ndcg_at_k(rels, 50),
        "map": average_precision(rels),
        "p@10": precision_at_k(rels, 10),
        "p@5": precision_at_k(rels, 5),
    }
    m["composite"] = (
        0.50 * m["ndcg@10"] + 0.30 * m["ndcg@50"]
        + 0.15 * m["map"] + 0.05 * m["p@10"]
    )
    return m
