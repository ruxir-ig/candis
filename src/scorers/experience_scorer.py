"""Experience fit. JD targets 5-9 years but explicitly considers strong
candidates outside the band, so we use a soft trapezoid: full credit in 5-9,
ramping up from ~3, and decaying (not zeroing) for very senior people.
"""
from ..config import interp

# (years, score) control points, linearly interpolated.
EXPERIENCE_CURVE = [
    (0, 0.00), (2, 0.25), (3, 0.55), (5, 1.00), (9, 1.00),
    (12, 0.70), (15, 0.40), (20, 0.25), (50, 0.15),
]


def experience_score(candidate: dict) -> float:
    yoe = candidate.get("profile", {}).get("years_of_experience", 0) or 0
    return round(interp(yoe, EXPERIENCE_CURVE), 4)
