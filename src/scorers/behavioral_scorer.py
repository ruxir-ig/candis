"""Behavioral availability in [0,1].

The JD: "a perfect-on-paper candidate who hasn't logged in for 6 months and has
a 5% recruiter response rate is, for hiring purposes, not actually available.
Down-weight them appropriately." So this is an availability/reachability score,
applied by the ranker as a *multiplier* on fit — never as additive fit itself.
"""
from datetime import date, datetime

# Sub-weights sum to 1.0.
SIGNAL_WEIGHTS = {
    "recency": 0.25,
    "response_rate": 0.20,
    "open_to_work": 0.12,
    "notice_period": 0.12,
    "interview_completion": 0.10,
    "profile_completeness": 0.08,
    "verification": 0.07,
    "offer_acceptance": 0.06,
}


def _parse(d):
    try:
        return datetime.strptime(d, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _recency(last_active, ref: date) -> float:
    d = _parse(last_active)
    if d is None:
        return 0.0
    days = (ref - d).days
    return max(0.0, 1.0 - days / 180.0)  # linear decay over 6 months


def _notice(days) -> float:
    if days is None:
        return 0.4
    if days <= 30:
        return 1.0
    if days <= 60:
        return 0.75
    if days <= 90:
        return 0.5
    if days <= 120:
        return 0.35
    return 0.2


def behavioral_availability(candidate: dict, ref: date) -> float:
    s = candidate.get("redrob_signals", {}) or {}
    w = SIGNAL_WEIGHTS

    offer = s.get("offer_acceptance_rate", -1)
    offer_component = 0.5 if (offer is None or offer < 0) else offer  # neutral if no history

    verifications = sum(
        1 for k in ("verified_email", "verified_phone", "linkedin_connected") if s.get(k)
    ) / 3.0

    score = (
        w["recency"] * _recency(s.get("last_active_date"), ref)
        + w["response_rate"] * min(1.0, s.get("recruiter_response_rate", 0) or 0)
        + w["open_to_work"] * (1.0 if s.get("open_to_work_flag") else 0.0)
        + w["notice_period"] * _notice(s.get("notice_period_days"))
        + w["interview_completion"] * min(1.0, s.get("interview_completion_rate", 0) or 0)
        + w["profile_completeness"] * min(1.0, (s.get("profile_completeness_score", 0) or 0) / 100.0)
        + w["verification"] * verifications
        + w["offer_acceptance"] * offer_component
    )
    return round(min(1.0, score), 4)
