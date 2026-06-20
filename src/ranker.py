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

