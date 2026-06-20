"""Evidence Graph ranker — rewards corroboration across fields.

The per-field scorers (title, skills, career, ...) each read one slice of the
profile independently. A keyword-stuffer can score well on `skills` while their
career prose is pure HR. The evidence graph reads fields *jointly* and rewards
agreement: a candidate whose skills list FAISS *and* whose career description
says "built semantic search" *and* whose assessment backs it up is strongly
corroborated; one whose skills list FAISS but whose career text never mentions
search/retrieval is not.

Eight evidence categories (soft-scored, never hard-filtered — a sparse profile
isn't penalised to zero, just weakly corroborated):

  1. role_skill_corroboration  — retrieval/search in BOTH skills and career text
  2. retrieval_maturity        — career prose shows shipped ranking rigor
                                 (NDCG/MRR/MAP/A-B/LTR/recommendation/retrieval)
  3. assessment_support        — skill_assessment_scores back claimed ML skills
  4. duration_support          — key skills have real tenure (expert+0mo = red flag)
  5. seniority_consistency     — years of experience align with title seniority
  6. product_context           — product company + ML title + ML skills all agree
  7. behavioral_readiness      — referenced for explanation (scored elsewhere)
  8. anti_stuffing             — contradiction penalty (multiplier, not additive)

Output per candidate: {score, components, positive[], negative[]} — the evidence
strings feed the reasoning column.
"""
from __future__ import annotations

# --- concept term sets -------------------------------------------------------
# Retrieval/search concepts that should appear in BOTH skills and career prose
# for a genuine search/retrieval engineer.
RETRIEVAL_CONCEPTS = {
    "semantic search", "search relevance", "retrieval", "ranking",
    "recommendation", "recommender", "vector search", "nearest neighbor",
    "faiss", "qdrant", "pinecone", "weaviate", "milvus",
    "elasticsearch", "opensearch", "embeddings", "rag",
}

# Shipped-rigor terms: the vocabulary of someone who has OWNED ranking quality,
# not just touched a model. These in career prose are strong positive evidence.
MATURITY_TERMS = {
    "ndcg", "mrr", "map", "a/b test", "a/b testing", "ab test", "ab testing",
    "learning to rank", "search relevance", "recommendation system",
    "retrieval quality", "relevance tuning", "online metrics", "offline metrics",
    "recall@", "precision@", "hit rate", "conversion rate",
}

# Skills that, if claimed, most need career-text corroboration (the stuffer
# favourites).
HIGH_STAKES_SKILLS = {
    "faiss", "qdrant", "pinecone", "weaviate", "milvus", "opensearch",
    "elasticsearch", "semantic search", "vector search", "vector database",
    "rag", "retrieval", "embeddings", "sentence transformers",
    "recommendation system", "learning to rank",
}

SENIOR_MARKERS = ("staff", "principal", "head", "lead", "senior", "architect")
# Minimum years plausibly needed for a senior-ish title (soft, not a hard gate).
SENIOR_MIN_YEARS = 4.0


def _career_text(candidate: dict) -> str:
    parts = []
    for r in candidate.get("career_history", []) or []:
        d = (r.get("description") or "") + " " + (r.get("title") or "")
        parts.append(d)
    p = candidate.get("profile", {})
    if p.get("summary"):
        parts.append(p["summary"])
    return " ".join(parts).lower()


def _skill_names(candidate: dict) -> list[str]:
    return [s.get("name", "") for s in candidate.get("skills", []) or [] if s.get("name")]


def _contains_any(haystack: str, terms: set) -> set:
    """Which terms appear in the haystack."""
    return {t for t in terms if t in haystack}


# --- individual evidence categories -----------------------------------------

def _role_skill_corroboration(career_text: str, skill_text: str) -> tuple[float, list[str]]:
    """High when retrieval/search concepts appear in BOTH skills and career."""
    in_skills = _contains_any(skill_text, RETRIEVAL_CONCEPTS)
    in_career = _contains_any(career_text, RETRIEVAL_CONCEPTS)
    if not in_skills and not in_career:
        return 0.5, []  # neutral — neither claims it (a backend eng is fine)
    agreed = in_skills & in_career
    if agreed:
        return 1.0, [f"Retrieval/search appears in both skills and career: "
                     f"{', '.join(sorted(agreed)[:4])}"]
    if in_skills and not in_career:
        return 0.15, []  # skill claims search but career text never shows it
    # career mentions it but not in skills list — still decent, prose is honest
    return 0.65, [f"Career describes retrieval work ({', '.join(sorted(in_career)[:3])})"]


def _retrieval_maturity(career_text: str) -> tuple[float, list[str]]:
    """Career prose shows shipped ranking rigor (evaluation metrics etc.)."""
    found = _contains_any(career_text, MATURITY_TERMS)
    if not found:
        return 0.3, []
    score = min(1.0, 0.55 + 0.15 * len(found))
    return score, [f"Career shows ranking rigor: {', '.join(sorted(found)[:4])}"]


def _assessment_support(candidate: dict) -> tuple[float, list[str]]:
    assessments = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {}) or {}
    if not assessments:
        return 0.4, []  # no assessments — neutral, not punitive
    relevant = {k: v for k, v in assessments.items()
                if k.lower() in HIGH_STAKES_SKILLS or any(t in k.lower() for t in RETRIEVAL_CONCEPTS)}
    if not relevant:
        # still credit any decent assessment as general ML competence
        good = [v for v in assessments.values() if v >= 60]
        return (0.55 if good else 0.4, [f"Assessments present ({len(good)} >=60)"])
    strong = [v for v in relevant.values() if v >= 60]
    score = min(1.0, 0.6 + 0.2 * len(strong))
    return score, [f"Assessment corroborates: {', '.join(f'{k}={v:.0f}' for k,v in relevant.items() if v>=60)[:60]}"]


def _duration_support(candidate: dict) -> tuple[float, list[str], list[str]]:
    """Key skills have real tenure. Flag expert+0months as contradiction."""
    skills = candidate.get("skills", []) or []
    pos, neg = [], []
    key = [s for s in skills if any(h in (s.get("name", "").lower())
                                    for h in HIGH_STAKES_SKILLS)]
    if not key:
        return 0.5, [], []
    contradictions = 0
    real_tenure = 0
    for s in key:
        prof = (s.get("proficiency") or "").lower()
        dur = s.get("duration_months")
        if prof == "expert" and (dur is None or dur == 0):
            contradictions += 1
            neg.append(f"'{s.get('name')}' claimed expert with ~0 months tenure")
        elif dur and dur >= 12:
            real_tenure += 1
    score = max(0.0, 0.7 + 0.1 * min(real_tenure, 3) - 0.3 * contradictions)
    if real_tenure:
        pos.append(f"{real_tenure} key retrieval skill(s) with 12+ months tenure")
    return round(score, 4), pos, neg


def _seniority_consistency(candidate: dict) -> tuple[float, list[str], list[str]]:
    """Years of experience vs title seniority. Staff with 2yr = inflation."""
    p = candidate.get("profile", {})
    years = p.get("years_of_experience") or 0
    title = (p.get("current_title") or "").lower()
    pos, neg = [], []
    is_senior = any(m in title for m in SENIOR_MARKERS)
    if not is_senior:
        return 0.6, [], []
    if years >= SENIOR_MIN_YEARS:
        return 1.0, [f"Senior title '{p.get('current_title','')}' matches {years:.1f}y experience"], []
    # senior title but thin experience
    score = max(0.2, 0.5 - 0.1 * (SENIOR_MIN_YEARS - years))
    neg.append(f"Senior title but only {years:.1f}y experience (possible inflation)")
    return round(score, 4), pos, neg


def _product_context(candidate: dict) -> tuple[float, list[str]]:
    """Product company + ML title + ML skills all agree = strong context."""
    from .scorers.career_scorer import _is_services
    history = candidate.get("career_history", []) or []
    has_product = any(not _is_services(r.get("company", "")) for r in history)
    pos = []
    if has_product:
        companies = {r.get("company", "") for r in history if not _is_services(r.get("company", ""))}
        pos.append(f"Product-company experience: {', '.join(sorted(companies))[:50]}")
    return (0.85 if has_product else 0.35), pos


def _title_skill_corroboration(candidate: dict) -> tuple[float, list[str], list[str]]:
    """The title is the single strongest anti-stuffer signal. A non-technical
    title (Operations/HR/Sales) paired with a high ML-skill list is the classic
    keyword-stuffer footprint — penalise it hard. A real ML title corroborating
    ML skills is the strongest positive."""
    from .scorers import title_role_score, skills_score
    ts = title_role_score(candidate)
    ss = skills_score(candidate)
    pos, neg = [], []
    if ts < 0.30 and ss > 0.40:
        neg.append(f"Non-technical title but {ss:.0%} ML skills — keyword-stuffer footprint")
        return 0.05, pos, neg
    if ts >= 0.80 and ss > 0.40:
        pos.append("ML title strongly corroborates ML skills")
        return 1.0, pos, neg
    # smooth in between
    return round(0.35 + 0.65 * ts, 4), pos, neg


# --- composite ---------------------------------------------------------------

# Weights for the additive categories (anti_stuffing is a multiplier below).
EVIDENCE_WEIGHTS = {
    "title_skill_corroboration": 0.22,   # anchor against keyword-stuffer traps
    "role_skill_corroboration": 0.20,
    "retrieval_maturity": 0.16,
    "assessment_support": 0.12,
    "duration_support": 0.12,
    "seniority_consistency": 0.08,
    "product_context": 0.10,
}


def evidence_graph_score(candidate: dict) -> dict:
    """Return {score, components, positive[], negative[]} for one candidate."""
    career_text = _career_text(candidate)
    skill_text = " ".join(_skill_names(candidate)).lower()

    c0, p0, n0 = _title_skill_corroboration(candidate)
    c1, p1 = _role_skill_corroboration(career_text, skill_text)
    c2, p2 = _retrieval_maturity(career_text)
    c3, p3 = _assessment_support(candidate)
    c4, p4, n4 = _duration_support(candidate)
    c5, p5, n5 = _seniority_consistency(candidate)
    c6, p6 = _product_context(candidate)

    components = {
        "title_skill_corroboration": c0,
        "role_skill_corroboration": c1,
        "retrieval_maturity": c2,
        "assessment_support": c3,
        "duration_support": c4,
        "seniority_consistency": c5,
        "product_context": c6,
    }
    positive = [x for x in (p0 + p1 + p2 + p3 + p4 + p5 + p6) if x]
    negative = [x for x in (n0 + n4 + n5) if x]

    score = sum(EVIDENCE_WEIGHTS[k] * components[k] for k in EVIDENCE_WEIGHTS)

    # Anti-stuffing multiplier: if we accumulated explicit contradictions, damp.
    if negative:
        score *= max(0.3, 1.0 - 0.25 * len(negative))

    return {
        "score": round(max(0.0, min(1.0, score)), 4),
        "components": {k: round(v, 4) for k, v in components.items()},
        "positive": positive,
        "negative": negative,
    }


def evidence_phrase(candidate: dict) -> str:
    """The single most informative corroboration point, or '' if none.

    Used to enrich the submission's reasoning column with a *why* grounded in
    cross-field agreement (e.g. "skills + career both show retrieval"). Picks the
    maturity/corroboration evidence first — the signal no per-field scorer states."""
    e = evidence_graph_score(candidate)
    for key in ("Retrieval/search appears", "Career shows ranking rigor",
                "ML title strongly corroborates", "Assessment corroborates",
                "Career describes retrieval", "Product-company"):
        for p in e["positive"]:
            if p.startswith(key):
                return p
    return ""
