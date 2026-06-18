"""Stage 5: combine all scorers into one ranking over the full pool.

score_all() returns every (non-honeypot) candidate scored and sorted. rank.py
takes the top 100; the eval harness restricts the same ordering to the labeled
golden set. Keeping one scoring path means what we evaluate is exactly what we
submit.
"""
from datetime import date, datetime
from pathlib import Path

from .config import (
    FIT_WEIGHTS, FIT_WEIGHTS_HYBRID, BEHAVIORAL_FLOOR, DROP_HONEYPOTS,
    REFERENCE_DATE_FALLBACK, SCORE_DECIMALS,
)
from .honeypot_detector import is_honeypot
from .coarse_filter import passes_coarse_filter
from .scorers import (
    title_role_score, skills_score, career_score, experience_score,
    education_score, location_score, behavioral_availability,
)
from .semantic_matcher import semantic_scores

ROOT = Path(__file__).resolve().parent.parent
SPARSE_CACHE = ROOT / "cache" / "sparse_scores.npz"


def reference_date(candidates) -> date:
    """Use the latest last_active_date in the data as 'today' for recency math."""
    latest = None
    for c in candidates:
        d = c.get("redrob_signals", {}).get("last_active_date")
        try:
            parsed = datetime.strptime(d, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
        if latest is None or parsed > latest:
            latest = parsed
    return latest or REFERENCE_DATE_FALLBACK


def score_candidate(candidate: dict, ref: date, weights: dict, sem: dict) -> dict:
    components = {
        "title": title_role_score(candidate),
        "skills": skills_score(candidate),
        "career": career_score(candidate),
        "experience": experience_score(candidate),
        "education": education_score(candidate),
        "location": location_score(candidate),
    }
    if "semantic" in weights:
        components["semantic"] = sem.get(candidate["candidate_id"], 0.0) if sem else 0.0

    fit = sum(weights[k] * components[k] for k in weights)
    availability = behavioral_availability(candidate, ref)
    final = fit * (BEHAVIORAL_FLOOR + (1 - BEHAVIORAL_FLOOR) * availability)

    return {
        "candidate_id": candidate["candidate_id"],
        "score": round(final, SCORE_DECIMALS),
        "fit": round(fit, 4),
        "availability": availability,
        "components": components,
        "candidate": candidate,
    }


def score_all(candidates, ref: date = None, drop_honeypots: bool = DROP_HONEYPOTS):
    if ref is None:
        ref = reference_date(candidates)

    # Use hybrid weights iff the embedding cache is available; else rule-only.
    sem = semantic_scores()
    weights = FIT_WEIGHTS_HYBRID if sem is not None else FIT_WEIGHTS
    mode = "hybrid (rule + semantic)" if sem is not None else "rule-only (no embedding cache)"
    print(f"  ranking mode: {mode}")

    # Two-stage hard filter: honeypots (logically impossible) then coarse domain
    # filter (full-career non-technical = keyword-stuffer safety net). Both are
    # hard guarantees; the soft scorers below can never override them.
    kept = []
    honeypots = coarse_dropped = 0
    for c in candidates:
        if drop_honeypots and is_honeypot(c):
            honeypots += 1
            continue
        if not passes_coarse_filter(c):
            coarse_dropped += 1
            continue
        kept.append(c)
    if honeypots or coarse_dropped:
        print(f"  hard filter: {honeypots} honeypots, {coarse_dropped} non-technical careers removed")

    scored = [score_candidate(c, ref, weights, sem) for c in kept]
    # Round first, then sort by (-score, candidate_id): guarantees the
    # validator's tie-break rule (equal scores -> candidate_id ascending).
    scored.sort(key=lambda r: (-r["score"], r["candidate_id"]))
    return scored, ref


def load_sparse_scores():
    """Load cached BM25 sparse scores. None if the cache is absent (so the
    runtime falls back to the weighted-fit ensemble alone)."""
    if not SPARSE_CACHE.exists():
        return None
    import numpy as np
    d = np.load(SPARSE_CACHE, allow_pickle=True)
    return {str(cid): float(s) for cid, s in zip(d["candidate_ids"], d["scores"])}


def apply_rrf(scored: list, sparse_scores: dict, k: int = 60,
              sparse_weight: float = 1.0) -> list:
    """Re-order `scored` (the weighted-fit+availability ranking) by Reciprocal
    Rank Fusion with the BM25 sparse ranking.

    Rank list A = the ensemble order (already sorted by final score).
    Rank list B = BM25 order. The fused RRF score replaces each row's `score`
    so downstream stages (LLM re-rank, CSV writer) see the fused ordering. The
    rich per-candidate dicts (components, fit, candidate) are preserved.

    `sparse_weight` down-weights BM25's vote without dropping it — useful if BM25
    proves noisy (the plan's "use BM25 only as a weak ranker in RRF" lever).
    """
    from .fusion import reciprocal_rank_fusion
    list_a = [r["candidate_id"] for r in scored]
    list_b = [cid for cid, _ in sorted(sparse_scores.items(),
                                       key=lambda kv: (-kv[1], kv[0]))]
    rrf, order = reciprocal_rank_fusion(
        [("ensemble", list_a), ("sparse", list_b)],
        k=k, weights={"sparse": sparse_weight})
    by_id = {r["candidate_id"]: r for r in scored}
    out = []
    for cid in order:
        r = dict(by_id[cid])
        r["score"] = round(float(rrf[cid]), 6)
        out.append(r)
    return out
