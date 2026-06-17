"""Career quality: product-company experience, progression, tenure stability.

The JD is explicit: people whose entire career is at IT-services/consulting
firms are "not a fit", and title-chasers who hop every ~1.5 years are a
negative. We can't perfectly label "product company" without a registry, so we
treat "not in the known services list" as a weak proxy for product, and lean on
progression + tenure as corroborating signals.
"""

SERVICES_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mindtree", "ltimindtree", "lti",
    "mphasis", "dxc", "syntel", "igate", "hexaware", "birlasoft", "coforge",
    "persistent systems", "zensar", "cybage", "nttdata", "ntt data",
    "deloitte", "pwc", "kpmg", "ey", "ernst", "capco",
}

SENIOR_MARKERS = ("senior", "lead", "principal", "staff", "head", "architect")


def _is_services(company: str) -> bool:
    c = (company or "").lower()
    return any(s in c for s in SERVICES_COMPANIES)


def is_services_only(candidate: dict) -> bool:
    history = candidate.get("career_history", []) or []
    if not history:
        return False
    return all(_is_services(r.get("company", "")) for r in history)


def career_score(candidate: dict) -> float:
    history = candidate.get("career_history", []) or []
    if not history:
        return 0.4

    score = 0.6

    has_product = any(not _is_services(r.get("company", "")) for r in history)
    all_services = not has_product
    if has_product:
        score += 0.2
    if all_services:
        score -= 0.3

    # Progression: a senior marker appears in a recent role but not the earliest.
    titles = [(r.get("title", "") or "").lower() for r in history]
    recent_senior = any(m in t for t in titles[:2] for m in SENIOR_MARKERS)
    early_senior = any(m in titles[-1] for m in SENIOR_MARKERS)
    if recent_senior and not early_senior:
        score += 0.1

    # Job-hopping: several short non-current stints reads as title-chasing.
    short_stints = sum(
        1 for r in history
        if not r.get("is_current") and (r.get("duration_months") or 0) < 14
    )
    if short_stints >= 3:
        score -= 0.15

    return round(max(0.0, min(1.0, score)), 4)
