"""Serialize a candidate into a compact, token-efficient text block for the LLM.

Token cost matters: we pay per token and we make many calls. So we keep only
the fields an expert recruiter actually weighs, in priority order, and we trim
verbose free-text to its leading signal. This is the *judge view* of a
candidate — deliberately smaller than the embedding view (profile_text.py).
"""
from __future__ import annotations


def _skill_line(s: dict) -> str:
    prof = s.get("proficiency", "?")
    months = s.get("duration_months", 0)
    assessed = s.get("skill_assessment_score")
    tag = f" (assessed {assessed})" if assessed not in (None, "", 0) else ""
    return f"{s.get('name','?')} [{prof}, {months}mo{tag}]"


def candidate_brief(candidate: dict, max_skills: int = 24, max_roles: int = 4) -> str:
    """A compact recruiter-style brief of the candidate."""
    prof = candidate.get("profile", {})
    sig = candidate.get("redrob_signals", {})
    lines = []

    title = prof.get("current_title") or ""
    company = prof.get("current_company") or ""
    yoe = prof.get("years_of_experience")
    lines.append(f"CURRENT: {title} @ {company} | {yoe} yrs exp".strip())

    loc = prof.get("location") or ""
    if loc:
        lines.append(f"LOCATION: {loc}")

    head = prof.get("headline") or ""
    summ = (prof.get("summary") or "").strip()
    if head:
        lines.append(f"HEADLINE: {head}")
    if summ:
        lines.append(f"SUMMARY: {summ[:500]}")

    skills = candidate.get("skills") or []
    if skills:
        shown = skills[:max_skills]
        lines.append("SKILLS: " + "; ".join(_skill_line(s) for s in shown))

    hist = candidate.get("career_history") or []
    if hist:
        lines.append("CAREER:")
        for r in hist[:max_roles]:
            cur = " (current)" if r.get("is_current") else ""
            dur = r.get("duration_months", 0)
            comp = r.get("company", "")
            t = r.get("title", "")
            line = f"  - {t} @ {comp}{cur} [{dur}mo]"
            desc = (r.get("description") or "").strip()
            if desc:
                line += f" :: {desc[:220]}"
            lines.append(line)

    edu = candidate.get("education") or []
    if edu:
        e = edu[0]
        lines.append(f"EDUCATION: {e.get('degree','')} {e.get('field_of_study','')} "
                     f"@ {e.get('institution','')} [{e.get('tier','')}]")

    # Behavioral / availability signals — the things that make someone
    # actually hireable vs just-good-on-paper.
    if sig:
        lines.append(
            f"SIGNALS: response_rate={sig.get('recruiter_response_rate')} | "
            f"notice={sig.get('notice_period_days')}d | "
            f"last_active={sig.get('last_active_date')} | "
            f"profile_complete={sig.get('profile_completeness_score')}/100 | "
            f"interview_complete={sig.get('interview_completion_rate')}"
        )

    return "\n".join(lines)
