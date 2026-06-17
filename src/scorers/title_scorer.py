"""Title / role fit.

This is the single strongest signal against the dataset's keyword-stuffer trap:
a "Marketing Manager" who lists 9 AI skills must score low here no matter how
good the skill list looks. We score the current title heavily, but also give
partial credit for the best recent role (someone who *just* moved out of an ML
role is still plausible).
"""

# Ordered, first-match-wins. Strong roles are listed before negatives so a
# genuine "ML Engineer" is never caught by a stray word; negatives are listed
# before the generic "engineer"/"analyst" catch-alls so non-technical managers
# don't get generic credit.
TITLE_RULES = [
    # Strong, directly-relevant IC roles
    (1.00, "machine learning engineer"), (1.00, "ml engineer"),
    (1.00, "ai engineer"), (1.00, "artificial intelligence engineer"),
    (1.00, "applied scientist"), (1.00, "applied ml"),
    (1.00, "nlp engineer"), (1.00, "search engineer"),
    (1.00, "relevance engineer"), (1.00, "search relevance"),
    (1.00, "recommendation"), (1.00, "recsys"),
    (1.00, "information retrieval"), (0.95, "deep learning engineer"),
    (0.95, "ml scientist"), (0.95, "machine learning scientist"),
    (0.92, "research engineer"), (0.90, "machine learning"),
    (0.85, "mlops"), (0.85, "ml platform"),
    # Data science: relevant but can drift toward analytics
    (0.85, "data scientist"), (0.80, "ml researcher"),
    # Non-technical / wrong-domain titles (the traps) -> near zero
    (0.05, "marketing"), (0.05, "sales"), (0.05, "human resources"),
    (0.05, "hr "), (0.05, "content writer"), (0.05, "copywriter"),
    (0.05, "accountant"), (0.10, "finance"), (0.10, "recruiter"),
    (0.10, "talent acquisition"), (0.10, "designer"), (0.10, "ux"),
    (0.10, "customer"), (0.12, "operations"), (0.15, "project manager"),
    (0.15, "program manager"), (0.20, "product manager"),
    (0.20, "business analyst"), (0.20, "qa"), (0.20, "tester"),
    (0.20, "teacher"), (0.20, "professor"), (0.30, "consultant"),
    # Adjacent engineering: plausible with other signals
    (0.65, "data engineer"), (0.65, "backend engineer"),
    (0.62, "platform engineer"), (0.60, "software engineer"),
    (0.58, "full stack"), (0.58, "software developer"),
    (0.55, "sde"), (0.55, "computer scientist"),
    # Generic catch-alls
    (0.40, "engineer"), (0.40, "developer"), (0.38, "architect"),
    (0.38, "programmer"), (0.35, "analyst"),
]

DEFAULT_TITLE_SCORE = 0.15


def score_title_text(title: str) -> float:
    if not title:
        return DEFAULT_TITLE_SCORE
    t = title.lower()
    for score, kw in TITLE_RULES:
        if kw in t:
            return score
    return DEFAULT_TITLE_SCORE


def title_role_score(candidate: dict) -> float:
    """0.7 * current title + 0.3 * best title seen in career history."""
    profile = candidate.get("profile", {})
    current = score_title_text(profile.get("current_title", ""))

    history = candidate.get("career_history", [])
    best_hist = max(
        (score_title_text(r.get("title", "")) for r in history),
        default=current,
    )
    return round(0.7 * current + 0.3 * best_hist, 4)
