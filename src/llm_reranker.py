"""LLM re-ranker — the final, highest-leverage stage (offline-cached).

The rule + semantic ensemble produces a strong recall set. Then an LLM judge
re-orders the top window: it reads each shortlisted candidate's full brief
*jointly* with the role and grades it 0-4 — a pointwise cross-attention pass
that beats any bi-encoder cosine at telling a real ML engineer from a
keyword-stuffer. That re-ordering directly shapes NDCG@10, which is half the
score.

The grades are produced OFFLINE (precompute/llm_label.py --sample topN) and
cached in cache/llm_rerank.jsonl. At ranking time we only load the cache and
re-sort — no network, no torch, fits the 5-min/CPU budget. If the cache is
missing or covers too little of the window, we fall back to the ensemble order
unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RERANK_CACHE = ROOT / "cache" / "llm_rerank.jsonl"

# Minimum fraction of the re-rank window that must be LLM-graded before we trust
# the re-order (avoids a half-graded window reshuffling badly).
MIN_COVERAGE = 0.9


def load_grades(path: Path = RERANK_CACHE) -> dict:
    """{candidate_id: fit_score} from the cached LLM re-rank run. Empty if absent.

    Uses the continuous fit_score (0-10), not the integer grade: the top-200 is
    mostly grade-4, so the integer grade can't discriminate among them and the
    re-rank becomes a no-op. The continuous score is what makes re-ordering work.
    """
    p = Path(path)
    if not p.exists():
        return {}
    out = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            fs = d.get("fit_score")
            if fs is None:
                continue
            try:
                out[d["candidate_id"]] = float(fs)
            except (TypeError, ValueError):
                continue
    return out


def apply_rerank(scored: list, window: int = 200, grades: dict | None = None) -> list:
    """Re-sort the top `window` of `scored` by (LLM fit_score desc, fit desc).

    Candidates outside the window keep their ensemble order. fit_score is the
    LLM's continuous 0-10 quality judgment (primary key); the ensemble fit score
    is only a tiebreak. If too few of the window is graded, return unchanged.
    """
    if grades is None:
        grades = load_grades()
    if not grades or window <= 0:
        return scored

    head = scored[:window]
    covered = sum(1 for r in head if r["candidate_id"] in grades)
    if covered / len(head) < MIN_COVERAGE:
        return scored  # not enough LLM coverage to trust a reshuffle

    tail = scored[window:]
    # Primary key: LLM continuous fit_score (ungraded -> -1, sinks in window).
    # Tiebreak: the ensemble fit score (keeps availability in the picture).
    head_sorted = sorted(
        head,
        key=lambda r: (grades.get(r["candidate_id"], -1.0), r["score"]),
        reverse=True,
    )

    # The displayed score must be strictly decreasing down the final list (the
    # validator rejects equal scores not ordered by candidate_id). So assign the
    # re-ranked window strictly-decreasing positional scores that sit in a band
    # ABOVE the tail's top. The ORDER already encodes the LLM judgment; the score
    # is only an ordering device (the challenge scores on ORDER via NDCG, not on
    # score values), and the reasoning column carries the 'why'.
    tail_top = tail[0]["score"] if tail else 0.0
    hi, lo = 1.0, tail_top + 1e-4
    n = len(head_sorted)
    step = (hi - lo) / n if n else 0.0
    for i, r in enumerate(head_sorted):
        r["score"] = round(hi - i * step, 6)
    return head_sorted + tail
