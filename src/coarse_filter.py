"""Stage 2: coarse hard-filter — a safety net, not the primary signal.

The soft scorers already penalize wrong-domain candidates (title scorer gives a
Marketing Manager ~0.05). So this filter exists for one reason only: a HARD
guarantee that the keyword-stuffer trap the JD warns about can NEVER reach the
top-100, even if a profile somehow games the skills + semantic layers.

Rule: if EVERY role in a candidate's career history is an unambiguous
non-technical domain (marketing / sales / HR / recruiting / accounting /
finance / content / design / customer support / teaching), they are rejected
outright. A single technical role anywhere in the history is enough to pass —
we filter only fully non-technical careers, and missing/empty history always
passes (never hard-reject on absent data).

We deliberately do NOT hard-cut on experience or location: the JD explicitly
says it considers strong candidates outside the 5-9yr band, and non-India is a
soft preference. Those stay soft in the scorers.
"""

# Unambiguous non-technical domains. Substring match on the lowercased title.
# Keep this narrower than the title_scorer's negatives: a hard reject must be
# defensible, so we only list domains where "this person is not an engineer,
# full stop" is essentially certain.
NON_TECHNICAL_MARKERS = (
    "marketing", "sales", "account executive", "account manager",
    "business development", "human resources", "hr manager", "hr executive",
    "talent acquisition", "recruiter", "recruiting",
    "accountant", "accounts", "finance manager", "financial analyst",
    "content writer", "copywriter", "technical writer",
    "graphic designer", "ux designer", "ui designer", "visual designer",
    "customer support", "customer service", "call centre", "call center",
    "teacher", "lecturer", "professor",  # teaching roles w/o eng in title
    "admin", "administrator", "office manager",
    "legal", "lawyer", "paralegal",
    "nurse", "doctor", "physician", "pharmacist",
)


def is_non_technical_title(title: str) -> bool:
    t = (title or "").lower()
    if not t:
        return False  # unknown title -> don't treat as evidence of non-tech
    return any(m in t for m in NON_TECHNICAL_MARKERS)


def passes_coarse_filter(candidate: dict) -> bool:
    """True unless the candidate's ENTIRE career is non-technical."""
    history = candidate.get("career_history") or []
    if not history:
        return True  # never hard-reject on missing data
    return any(not is_non_technical_title(r.get("title", "")) for r in history)
