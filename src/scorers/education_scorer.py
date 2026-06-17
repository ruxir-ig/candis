"""Education fit — institution tier x field relevance.

The JD doesn't emphasize education, so this stays a small-weight tiebreaker
(0.05 in the plan). But tier_1 (IIT / IISc / Stanford / CMU / BITS / NIT) with
a relevant field (CS / ML / AI / Data Science) is a real positive signal that
separates otherwise-equal candidates. We score the BEST education entry, not
the average — a B.Sc from a tier_3 college followed by an M.Tech at IIT should
land on the IIT signal.
"""

# Internal prestige tiering provided in the dataset.
TIER_SCORE = {
    "tier_1": 1.00,   # IIT, IISc, IIIT-H, BITS, NIT, Stanford, CMU, MIT, ...
    "tier_2": 0.70,
    "tier_3": 0.45,
    "tier_4": 0.25,
}

# Field-of-study relevance to a Senior AI/ML / search-ranking role.
# Strong = directly the discipline; adjacent = quantitative/engineering base
# that transitions plausibly; everything else is treated as neutral-low.
STRONG_FIELDS = {
    "computer science", "computer engineering", "computer sci",
    "artificial intelligence", "machine learning", "data science",
    "information technology", "software engineering",
}
ADJACENT_FIELDS = {
    "statistics", "mathematics", "applied mathematics", "electronics",
    "electronics & communication", "electrical engineering", "physics",
    "data engineering",
}
FIELD_FACTOR = {  # resolved per-entry; default 0.6 for unrelated (MBA, Civil, ...)
    "strong": 1.00,
    "adjacent": 0.80,
    "other": 0.60,
}

ADVANCED_DEGREES = {"ph.d", "phd", "doctorate", "m.tech", "mtech", "m.e.", "me",
                    "m.s.", "ms", "m.sc", "msc", "postgraduate"}


def _field_factor(field: str) -> float:
    f = (field or "").strip().lower()
    if not f:
        return FIELD_FACTOR["other"]
    if any(s in f for s in STRONG_FIELDS):
        return FIELD_FACTOR["strong"]
    if any(s in f for s in ADJACENT_FIELDS):
        return FIELD_FACTOR["adjacent"]
    return FIELD_FACTOR["other"]


def _entry_score(entry: dict) -> float:
    tier = TIER_SCORE.get((entry.get("tier") or "").lower().strip())
    if tier is None:
        return 0.0  # missing/unknown tier -> contributes nothing
    base = tier * _field_factor(entry.get("field_of_study", ""))
    # Small bonus for an advanced degree in a strong/adjacent field — the JD
    # values research depth (eval frameworks, retrieval theory).
    degree = (entry.get("degree") or "").lower()
    ff = _field_factor(entry.get("field_of_study", ""))
    if any(d in degree for d in ADVANCED_DEGREES) and ff >= FIELD_FACTOR["adjacent"]:
        base += 0.08
    return base


def education_score(candidate: dict) -> float:
    edu = candidate.get("education") or []
    if not edu:
        return 0.3  # unknown education: mild neutral, never a strong positive
    best = max(_entry_score(e) for e in edu)
    return round(max(0.0, min(1.0, best)), 4)
