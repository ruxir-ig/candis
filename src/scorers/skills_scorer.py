"""Skills fit, trust-weighted.

The trap: a profile can list "FAISS / Pinecone / embeddings" as *expert* with 0
months of real use. So a raw keyword count is exactly what the JD warns against.
We weight each relevant skill by a trust factor = proficiency x duration, and
cross-check against the platform's objective skill_assessment_scores. A stuffed
skill (high proficiency, ~0 duration, no assessment) contributes almost nothing.
"""

# Relevance weight per skill family (keyword -> weight). Matched by substring on
# the lowercased skill name; the max matching weight wins.
SKILL_RELEVANCE = {
    # core retrieval / embeddings
    "embedding": 1.0, "sentence-transformer": 1.0, "sentence transformer": 1.0,
    "retrieval": 1.0, "semantic search": 1.0, "rag": 0.95, "bge": 1.0, "e5": 0.9,
    # vector databases / search infra
    "faiss": 1.0, "pinecone": 1.0, "weaviate": 1.0, "qdrant": 1.0, "milvus": 1.0,
    "elasticsearch": 0.9, "opensearch": 0.9, "vector": 0.9, "bm25": 0.85,
    # ranking / recsys / evaluation
    "learning to rank": 0.95, "ranking": 0.95, "recommendation": 0.95,
    "recsys": 0.95, "ndcg": 0.8, "mrr": 0.8, "a/b test": 0.7, "ab test": 0.7,
    # nlp / ir
    "nlp": 0.8, "natural language": 0.8, "information retrieval": 1.0,
    # llm / fine-tuning (nice-to-have)
    "llm": 0.7, "fine-tun": 0.7, "lora": 0.65, "qlora": 0.65, "peft": 0.65,
    "transformer": 0.7, "hugging": 0.6,
    # general ml / python eng
    "pytorch": 0.6, "tensorflow": 0.55, "scikit": 0.55, "xgboost": 0.6,
    "python": 0.5,
    # supportive data-eng (adjacent, modest)
    "spark": 0.35, "airflow": 0.3, "kafka": 0.3, "data pipeline": 0.35,
    "sql": 0.25,
}

# Half-saturation constant: ~this much summed trust-weighted relevance -> 0.5.
SKILL_SATURATION = 4.0

PROFICIENCY_WEIGHT = {
    "beginner": 0.30, "intermediate": 0.60, "advanced": 0.85, "expert": 1.00,
}


def _relevance(skill_name: str) -> float:
    n = (skill_name or "").lower()
    best = 0.0
    for kw, w in SKILL_RELEVANCE.items():
        if kw in n and w > best:
            best = w
    return best


def _trust(skill: dict, assessment: dict) -> float:
    prof = PROFICIENCY_WEIGHT.get(skill.get("proficiency", ""), 0.5)
    dur = skill.get("duration_months")
    # Missing duration is treated as moderate, not zero; but 0 months (the
    # stuffer signal) genuinely zeroes the contribution.
    dur_trust = 0.5 if dur is None else min(1.0, dur / 24.0)
    t = prof * dur_trust
    # Objective cross-check: a high platform assessment score boosts trust.
    score = assessment.get(skill.get("name", ""))
    if score is not None and score >= 60:
        t = min(1.0, t * 1.2)
    return t


def _contributions(candidate: dict):
    skills = candidate.get("skills", []) or []
    assessment = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {}) or {}
    out = []
    for s in skills:
        rel = _relevance(s.get("name", ""))
        if rel <= 0:
            continue
        out.append((s.get("name", ""), rel * _trust(s, assessment)))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def skills_score(candidate: dict) -> float:
    total = sum(c for _, c in _contributions(candidate))
    return round(total / (total + SKILL_SATURATION), 4)


def matched_skill_names(candidate: dict, k: int = 3):
    """Top relevant skill names actually present — used for grounded reasoning."""
    return [name for name, _ in _contributions(candidate)[:k] if name]
