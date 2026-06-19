"""Evidence-guided LLM expansion promotion (pi's two-gate policy).

The LLM re-rank only grades the ensemble's top-200. Genuinely strong candidates
at ranks 201-800 — buried by availability dampening, not by weak fit — never get
judged. The expansion grades a diverse, evidence-rich slice of that band
(cache/llm_expansion.jsonl, same model as the base cache for comparability) and
promotes the ones that are *clearly* better than current incumbents.

Two-gate policy (the instructor's spec):
  - top-20 FROZEN (never touch the saturated, score-critical zone)
  - elite band (21-50): fit >= rank50_thr + 0.20, grade 4, evidence >= 0.85,
    confidence high, no concerns
  - normal band (51-100): fit >= rank100_thr + 0.15, grade 4, evidence >= 0.75,
    confidence medium/high, no major concerns
  - every promotion displaces the weakest current incumbent in its band

The score column is reassigned strictly-decreasing down the list (validator-safe),
identical to the LLM re-rank's positional scheme — the ORDER is what matters.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPANSION_CACHE = ROOT / "cache" / "llm_expansion.jsonl"

# gate constants
MARGIN_ELITE = 0.20
MARGIN_NORMAL = 0.15
MIN_ELITE_FIT = 9.2
MIN_NORMAL_FIT = 8.8
MIN_ELITE_EVIDENCE = 0.85
MIN_NORMAL_EVIDENCE = 0.75
FROZEN_TOP = 20


def load_expansion_grades(path: Path = EXPANSION_CACHE) -> dict:
    """{candidate_id: {fit_score, grade, confidence, concerns, reasoning}}."""
    p = Path(path)
    if not p.exists():
        return {}
    out = {}
    for line in open(p, encoding="utf-8"):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("fit_score") is not None:
            out[d["candidate_id"]] = d
    return out


def _has_concern(d: dict) -> bool:
    conf = (d.get("confidence") or "").lower()
    if conf == "low":
        return True
    c = str(d.get("concerns") or "").lower()
    # only flag substantive concerns, not empty/"none"
    return bool(c) and c not in ("none", "n/a", "no concerns", "null")


def select_promotions(scored, base_grades, exp_grades, evidence_fn):
    """Return (elite, normal) promotion lists per the two-gate policy.

    scored: the LLM-re-ranked list (post apply_rerank).
    base_grades/exp_grades: {cid: grade_dict} from the two caches.
    evidence_fn: callable(candidate_dict) -> float.
    """
    # thresholds from current incumbents' cached fit_scores
    top50_fits = [base_grades[r["candidate_id"]]["fit_score"]
                  for r in scored[:50] if r["candidate_id"] in base_grades]
    top100_fits = [base_grades[r["candidate_id"]]["fit_score"]
                   for r in scored[:100] if r["candidate_id"] in base_grades]
    rank50_thr = min(top50_fits) if top50_fits else 0.0
    rank100_thr = min(top100_fits) if top100_fits else 0.0

    elite_fit = max(rank50_thr + MARGIN_ELITE, MIN_ELITE_FIT)
    normal_fit = max(rank100_thr + MARGIN_NORMAL, MIN_NORMAL_FIT)

    elite, normal = [], []
    base_ids = set(base_grades)  # already LLM-graded in the top-200 cache
    for cid, d in exp_grades.items():
        if cid in base_ids:  # already graded by the base cache
            continue
        if d.get("grade") != 4:
            continue
        fs = d.get("fit_score", 0)
        conf = (d.get("confidence") or "").lower()
        if _has_concern(d):
            continue
        ev = evidence_fn(cid)
        is_elite = (fs >= elite_fit and ev >= MIN_ELITE_EVIDENCE and conf == "high")
        is_normal = (fs >= normal_fit and ev >= MIN_NORMAL_EVIDENCE
                     and conf in ("medium", "high"))
        if is_elite:
            elite.append((cid, fs, ev, conf))
        elif is_normal:
            normal.append((cid, fs, ev, conf))

    elite.sort(key=lambda x: -x[1])
    normal.sort(key=lambda x: -x[1])
    return elite, normal, (rank50_thr, rank100_thr, elite_fit, normal_fit)


def apply_expansion(scored, base_grades, exp_grades, evidence_fn,
                    evidence_map=None):
    """Promote eligible expansion candidates into ranks 21-100; freeze top-20.

    Returns (new_scored, audit) where audit has 'entered' and 'left' lists.
    """
    elite, normal, thresholds = select_promotions(scored, base_grades, exp_grades, evidence_fn)
    promotions = {cid: fs for cid, fs, _, _ in elite + normal}
    promotion_band = {}  # cid -> 'elite' | 'normal'
    for cid, _, _, _ in elite:
        promotion_band[cid] = "elite"
    for cid, _, _, _ in normal:
        promotion_band[cid] = "normal"

    if not promotions:
        return scored, {"entered": [], "left": [], "thresholds": thresholds}

    by_id = {r["candidate_id"]: r for r in scored}
    # Find the candidate dicts for promoted ids (they're in the tail, ranks >200)
    promoted_rows = []
    for cid in promotions:
        if cid in by_id:
            promoted_rows.append(dict(by_id[cid]))

    # Build the new ordering:
    # 1. Frozen top-20 (unchanged)
    # 2. Merge current ranks 21-100 + promoted, sort by (fit_score desc, score desc)
    # 3. Displaced candidates + rest of tail in original order
    head = scored[:FROZEN_TOP]
    mid_incumbents = scored[FROZEN_TOP:100]

    def fit_of(r):
        cid = r["candidate_id"]
        if cid in promotions:
            return promotions[cid]
        return base_grades.get(cid, {}).get("fit_score", 0.0)

    merged = list(mid_incumbents) + promoted_rows
    merged.sort(key=lambda r: (-fit_of(r), -r["score"], r["candidate_id"]))

    # The top (100 - FROZEN_TOP) = 80 stay in ranks 21-100; the rest are displaced
    keep_n = 100 - FROZEN_TOP
    new_mid = merged[:keep_n]
    displaced = merged[keep_n:]
    # Promoted candidates originally came from the tail; remove their original
    # copies so full-list audits do not see duplicate candidate IDs. Top-100
    # order is unaffected because promotions are already in new_mid.
    tail = [r for r in scored[100:] if r["candidate_id"] not in promotions]

    # new ordering: head + new_mid + displaced + tail
    new_order_ids = ([r["candidate_id"] for r in head]
                     + [r["candidate_id"] for r in new_mid]
                     + [r["candidate_id"] for r in displaced]
                     + [r["candidate_id"] for r in tail])
    # rebuild scored list in new order, assign strictly-decreasing positional scores
    new_scored = []
    # head keeps its scores; we rescore from the top to guarantee monotonicity
    all_rows = {r["candidate_id"]: r for r in scored}
    # update promoted rows' metadata
    for r in promoted_rows:
        all_rows[r["candidate_id"]] = r

    hi = 1.0
    n = len(new_order_ids)
    step = hi / n
    for i, cid in enumerate(new_order_ids):
        r = dict(all_rows[cid])
        r["score"] = round(hi - i * step, 6)
        new_scored.append(r)

    entered = [cid for cid in promotions if cid in {r["candidate_id"] for r in new_mid}]
    new_top100_ids = {r["candidate_id"] for r in new_scored[:100]}
    left = [r["candidate_id"] for r in scored[:100] if r["candidate_id"] not in new_top100_ids]

    audit = {
        "entered": [(cid, promotion_band[cid], promotions[cid]) for cid in entered],
        "left": left,
        "thresholds": thresholds,
    }
    return new_scored, audit
