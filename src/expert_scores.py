"""Expert-mixture score decomposition.

Breaks the monolithic fit score into 7 specialized expert dimensions, each
answering one recruiter question:

1. ir_search       — Can they build retrieval / vector search / IR pipelines?
2. recsys_ranking  — Can they build recommendation / ranking / LTR systems?
3. nlp_llm         — Can they build NLP / LLM / RAG / transformer systems?
4. production_ml   — Have they shipped ML to production at scale?
5. seniority       — Are they senior enough (titles + tenure at product cos)?
6. availability    — Are they actually reachable (behavioral signals)?
7. anti_gaming     — Is their profile corroborated (not keyword-stuffed)?

Each dimension returns a 0-1 score. The audit compares per-dimension rankings
against the overall rank to find hidden gems and check top-100 dominance.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Dimension → skill keywords (reuses the relevance categories from skills_scorer
# but partitions them by expertise domain).
# ---------------------------------------------------------------------------

_DIM_SKILLS = {
    "ir_search": {
        "embedding", "sentence-transformer", "sentence transformer",
        "retrieval", "semantic search", "bge", "e5",
        "faiss", "pinecone", "weaviate", "qdrant", "milvus",
        "elasticsearch", "opensearch", "vector", "bm25",
        "information retrieval",
    },
    "recsys_ranking": {
        "learning to rank", "ranking", "recommendation",
        "recsys", "ndcg", "mrr", "a/b test", "ab test",
    },
    "nlp_llm": {
        "nlp", "natural language", "rag",
        "llm", "fine-tun", "lora", "qlora", "peft",
        "transformer", "hugging",
    },
    "production_ml": {
        "pytorch", "tensorflow", "scikit", "xgboost",
        "spark", "airflow", "kafka", "data pipeline", "python",
    },
}

# Titles / role keywords per dimension (substring match, lowercased)
_DIM_TITLES = {
    "ir_search": [
        "search", "retrieval", "ir", "information retrieval",
        "vector", "embedding",
    ],
    "recsys_ranking": [
        "recommendation", "recsys", "ranking", "rank",
    ],
    "nlp_llm": [
        "nlp", "natural language", "llm", "language",
        "ai engineer", "ai/ml",
    ],
    "production_ml": [
        "machine learning", "ml engineer", "data scientist",
        "applied scientist", "ml infra", "mlops",
    ],
    "seniority": [
        "senior", "staff", "principal", "lead", "director",
        "head", "chief",
    ],
}

# Services companies (career dilution)
SERVICES_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "ibm", "deloitte", "pwc", "ey",
    "genpact", "cognizant",
}

# Saturation constant (same logic as skills_scorer: half-saturation at ~4)
_SAT = 3.0

_PROFICIENCY = {
    "beginner": 0.30, "intermediate": 0.60,
    "advanced": 0.85, "expert": 1.00,
}


def _trust(skill: dict) -> float:
    prof = _PROFICIENCY.get(skill.get("proficiency", ""), 0.5)
    dur = skill.get("duration_months")
    dur_trust = 0.5 if dur is None else min(1.0, dur / 24.0)
    return prof * dur_trust


def _has_assessment(candidate: dict, skill_name: str) -> bool:
    scores = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    return any(
        skill_name.lower() in k.lower() and v >= 60
        for k, v in (scores or {}).items()
    )


def _skill_dim_score(candidate: dict, keywords: set[str]) -> float:
    """Trust-weighted skill score for one dimension."""
    total = 0.0
    for s in candidate.get("skills", []) or []:
        name = (s.get("name") or "").lower()
        if any(kw in name for kw in keywords):
            t = _trust(s)
            if _has_assessment(candidate, s.get("name", "")):
                t = min(1.0, t * 1.2)
            total += t
    return round(total / (total + _SAT), 4)


def _title_dim_score(candidate: dict, patterns: list[str]) -> float:
    """1.0 if the current or any past title matches a pattern, 0.0 otherwise.

    Career history titles count at half-weight (past role, not current).
    """
    prof = candidate.get("profile", {})
    current = (prof.get("current_title") or "").lower()
    score = 0.0
    for p in patterns:
        if p in current:
            score = 1.0
            break
    if score == 0.0:
        for role in candidate.get("career_history", []) or []:
            t = (role.get("title") or "").lower()
            for p in patterns:
                if p in t:
                    return 0.5
    return score


def _is_product_company(company: str) -> bool:
    c = (company or "").lower().strip()
    return c != "" and not any(s in c for s in SERVICES_COMPANIES)


def _production_signal(candidate: dict) -> float:
    """Evidence of shipping ML to production: descriptions mentioning scale/deploy."""
    score = 0.0
    for role in candidate.get("career_history", []) or []:
        desc = (role.get("description") or "").lower()
        company = role.get("company") or ""
        if any(w in desc for w in ("production", "deploy", "serving", "million",
                                    "billion", "scale", "pipeline", "served")):
            score += 0.3 if _is_product_company(company) else 0.1
    return min(1.0, score)


def _seniority_score(candidate: dict) -> float:
    """Seniority from title level + tenure at product companies."""
    prof = candidate.get("profile", {})
    title = (prof.get("current_title") or "").lower()
    yoe = prof.get("years_of_experience") or 0

    level = 0.0
    for kw, val in [("principal", 1.0), ("director", 1.0), ("head", 0.95),
                     ("chief", 1.0), ("staff", 0.9), ("lead", 0.8),
                     ("senior", 0.7)]:
        if kw in title:
            level = val
            break

    # Yoe band: ideal 5-9, still good 4-12
    if yoe >= 5 and yoe <= 9:
        yoe_s = 1.0
    elif yoe >= 4 and yoe <= 12:
        yoe_s = 0.8
    elif yoe >= 3:
        yoe_s = 0.5
    else:
        yoe_s = 0.2

    # Product company bonus
    company = (prof.get("current_company") or "").lower()
    co_bonus = 0.15 if _is_product_company(company) else 0.0

    return round(min(1.0, 0.4 * level + 0.4 * yoe_s + 0.2 + co_bonus), 4)


def _availability_score(candidate: dict, ref_date) -> float:
    """Reuse the behavioral scorer's availability."""
    from .scorers.behavioral_scorer import behavioral_availability
    return behavioral_availability(candidate, ref_date)


def _anti_gaming_score(candidate: dict) -> float:
    """Reuse the evidence graph's corroboration score."""
    from .evidence_graph import evidence_graph_score
    return evidence_graph_score(candidate)["score"]


def expert_scores(candidate: dict, ref_date=None) -> dict[str, float]:
    """Compute 7-dimensional expert scores for a candidate.

    Returns a dict with keys: ir_search, recsys_ranking, nlp_llm,
    production_ml, seniority, availability, anti_gaming.
    """
    s = {}
    for dim, keywords in _DIM_SKILLS.items():
        s[dim] = _skill_dim_score(candidate, keywords)

    # Blend skill + title for the 4 technical dimensions
    for dim in ("ir_search", "recsys_ranking", "nlp_llm", "production_ml"):
        title_s = _title_dim_score(candidate, _DIM_TITLES.get(dim, []))
        s[dim] = round(0.65 * s[dim] + 0.35 * title_s, 4)

    s["production_ml"] = round(
        0.5 * s["production_ml"] + 0.5 * _production_signal(candidate), 4
    )
    s["seniority"] = _seniority_score(candidate)
    s["availability"] = _availability_score(candidate, ref_date) if ref_date else 0.0
    s["anti_gaming"] = _anti_gaming_score(candidate)
    return s


def expert_profile_str(scores: dict[str, float]) -> str:
    """One-line visual profile: ###-- for each dimension."""
    bar = lambda v: "#" * round(v * 5) + "-" * (5 - round(v * 5))
    labels = ["ir_search", "recsys_ranking", "nlp_llm", "production_ml",
              "seniority", "availability", "anti_gaming"]
    return "  ".join(f"{l[:3]}:{bar(scores.get(l, 0))}" for l in labels)


DIMENSIONS = ["ir_search", "recsys_ranking", "nlp_llm", "production_ml",
              "seniority", "availability", "anti_gaming"]
