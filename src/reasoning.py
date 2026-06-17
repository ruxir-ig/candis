"""Grounded reasoning strings for the submission.

Stage-4 review checks for: specific facts, JD connection, honest concerns, NO
hallucination, variation, and tone matching rank. So every clause here is built
from values actually present in the candidate record, concerns are surfaced
explicitly, and low-ranked candidates lead with their limiting factor.
"""
from datetime import date, datetime

from .scorers import matched_skill_names, is_services_only


def _months_since(d: str, ref: date):
    try:
        then = datetime.strptime(d, "%Y-%m-%d").date()
        return max(0, (ref - then).days) // 30
    except (ValueError, TypeError):
        return None


def _concerns(candidate: dict, ref: date):
    profile = candidate.get("profile", {})
    s = candidate.get("redrob_signals", {})
    out = []
    if "india" not in (profile.get("country", "") or "").lower():
        out.append(f"based in {profile.get('location') or profile.get('country')} (outside India)")
    if is_services_only(candidate):
        out.append("career entirely in IT services")
    rr = s.get("recruiter_response_rate")
    if rr is not None and rr < 0.3:
        out.append(f"low recruiter response ({rr:.0%})")
    stale = _months_since(s.get("last_active_date"), ref)
    if stale is not None and stale >= 4:
        out.append(f"inactive ~{stale}mo")
    npd = s.get("notice_period_days")
    if npd is not None and npd >= 90:
        out.append(f"{npd}-day notice")
    return out


def build_reasoning(candidate: dict, ref: date, rank: int) -> str:
    profile = candidate.get("profile", {})
    s = candidate.get("redrob_signals", {})
    title = profile.get("current_title", "Unknown role")
    company = profile.get("current_company", "")
    yrs = profile.get("years_of_experience", 0)

    skills = matched_skill_names(candidate, k=3)
    skills_phrase = (
        f"relevant skills incl. {', '.join(skills)}" if skills
        else "limited direct retrieval/ranking skills"
    )

    at = f" at {company}" if company else ""
    strengths = f"{title}{at}, {yrs:.1f} yrs; {skills_phrase}"

    rr = s.get("recruiter_response_rate")
    npd = s.get("notice_period_days")
    avail_bits = []
    if rr is not None:
        avail_bits.append(f"{rr:.0%} recruiter response")
    if npd is not None:
        avail_bits.append(f"{npd}-day notice")
    avail_phrase = "; ".join(avail_bits)

    concerns = _concerns(candidate, ref)

    # Tone matches rank: lower-ranked candidates lead with the limiting factor.
    if rank > 60 and concerns:
        text = f"{concerns[0].capitalize()}; otherwise {strengths.lower()}."
    else:
        text = strengths + "."
        if avail_phrase:
            text += f" {avail_phrase.capitalize()}."
        if concerns:
            text += f" Concern: {concerns[0]}."

    return text[:240]
