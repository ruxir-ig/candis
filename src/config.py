"""Central configuration: every weight and threshold the ranker uses lives here.

Why one file: there is no leaderboard, so we tune by reading our own rankings
and our offline golden-set metrics. Keeping all knobs in one place makes that
tuning loop honest and reproducible instead of scattered magic numbers.
"""
from datetime import date

# --- Fit combination weights (sum to 1.0) -----------------------------------
# "Fit" = how well the candidate matches the role on paper. Behavioral
# availability is applied separately as a multiplier (see below): it dampens a
# good fit when someone is unreachable, but it never rescues a bad fit.
# Rule-only weights — used as the fallback when no embedding cache is present.
# Education is deliberately small (JD doesn't emphasize it; it's a tiebreaker).
FIT_WEIGHTS = {
    "title": 0.33,        # strongest defense against keyword-stuffer traps
    "skills": 0.25,       # relevant skills, trust-weighted by proficiency+duration
    "career": 0.16,       # product vs services, progression, tenure stability
    "experience": 0.11,   # 5-9 yr sweet spot
    "education": 0.05,    # institution tier x field relevance (tiebreaker)
    "location": 0.10,     # India preference (soft)
}

# Hybrid weights — used when cache/embeddings.npz exists. The semantic component
# reads profile prose (rule scorers read structured fields), so the two are
# complementary; title AND skills stay high to keep anchoring against keyword
# stuffers (semantic alone can be gamed by prose that just mentions AI a lot).
# Note: on our 53-label golden set the composite is saturated (flat at 0.977
# from sem=0.00 through sem=0.25), so this weight is a judgment call, not a
# tuned one — kept modest on purpose so trust-weighted skills stays the lead.
FIT_WEIGHTS_HYBRID = {
    "title": 0.28,
    "skills": 0.23,
    "career": 0.13,
    "experience": 0.09,
    "education": 0.05,
    "location": 0.07,
    "semantic": 0.15,
}

# final = fit * (BEHAVIORAL_FLOOR + (1 - BEHAVIORAL_FLOOR) * availability)
# floor 0.65 -> a totally unavailable candidate keeps 65% of their fit. We keep
# this fairly high on purpose: "moderately reachable" (e.g. 120-day notice) is
# not "unavailable", and over-dampening buries genuinely strong candidates.
BEHAVIORAL_FLOOR = 0.65

# Honeypots (logically impossible profiles) are removed from the top-100 pool.
DROP_HONEYPOTS = True

# Reference "today" for recency math. The ranker overrides this with the real
# max(last_active_date) found in the data; this is only a fallback.
REFERENCE_DATE_FALLBACK = date(2026, 6, 4)

# Output precision for the submission's score column.
SCORE_DECIMALS = 6


def interp(x, points):
    """Piecewise-linear interpolation. `points` is a sorted list of (x, y)."""
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0) if x1 != x0 else 0.0
            return y0 + t * (y1 - y0)
    return points[-1][1]
