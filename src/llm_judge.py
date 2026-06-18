"""LLM-as-judge prompt for candidate relevance grading.

Design choices that matter:
- The system prompt drills the ONE thing the JD warns about: keyword-stuffed
  non-technical profiles must score low regardless of their skill list. This is
  where small/mediocre models most often fail, so we anchor it hard.
- Few-shot: one textbook trap and one textbook star, with the *reasoning* that
  justifies each grade. This calibrates both the scale (what a 0 vs a 4 looks
  like) and the decision rule.
- Output is forced JSON so we can parse it at scale without regex fragility.
"""
from __future__ import annotations

from .candidate_brief import candidate_brief
from .jd_query import IDEAL_CANDIDATE_QUERY as IDEAL_CANDIDATE

JUDGE_SYSTEM = """You are a senior technical recruiter specializing in AI/ML hiring, evaluating candidates for ONE specific role.

The decisive principle for this role: the right candidate is NOT "whoever lists the most AI keywords." The dataset is deliberately seeded with trap profiles — people in non-technical roles (Marketing, HR, Sales, Accounting, Content) who paste "RAG / FAISS / Pinecone / embeddings" into their skills list. Grade them LOW no matter how many AI skills they list, because their actual career shows they have never built AI systems. The substance of the career (titles held, systems built, companies) outweighs the skills section.

Conversely, a candidate who genuinely built ranking / retrieval / recommendation / search systems at a product company is a strong fit even if they don't use the exact buzzwords.

Weigh, in order of importance:
1. Career substance — did they hold ML/Search/RecSys/AI engineering roles and actually ship ranking/retrieval/LLM systems? (strongest signal)
2. Skills with proof — skills backed by real duration + assessments count far more than a bare keyword.
3. Company quality — product companies > pure consulting/services (TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini-only careers are a poor fit).
4. Experience — ideal 5-9 yrs; strong candidates outside the band still count.
5. Behavioral availability — responsiveness, notice period, recency (affects hireability, not raw fit).
6. Education — minor tiebreaker (tier-1 CS institute is a small plus).

Grade on a 0-4 scale:
  4 = Excellent fit. Senior ML/Search engineer, shipped relevant systems at a product company, responsive.
  3 = Good fit. Plausibly strong, minor gaps.
  2 = Possible / borderline. Some relevant signal but real concerns.
  1 = Weak. Mostly missing the mark.
  0 = Not a fit / trap. Non-technical career, or keyword-stuffed with no substance.

ALSO give a continuous fit_score from 0.0 to 10.0 — this FINE-GRAINED score is the
important one, because many candidates share the same integer grade and it must
discriminate AMONG them (e.g. two grade-4 candidates: one who owns a production
retrieval pipeline at a top product company is a 9.5, another who is strong but
narrow is a 7.8). Use the full decimal range; don't cluster around round numbers.

Return STRICT JSON only: {"grade": <int 0-4>, "fit_score": <float 0-10>, "confidence": <"high"|"medium"|"low">, "reasoning": "<2-3 sentences citing specific facts from the profile>", "concerns": "<main risk, or 'none'>"}"""

# Few-shot anchors. Trap first (so the keyword-stuffer rule is front-loaded),
# then a star. Kept short to control token cost across many calls.
_FEWSHOT_TRAP = """Marketing Manager @ TCS | 6.1 yrs exp
SKILLS: RAG [expert, 0mo]; FAISS [expert, 0mo]; Pinecone [expert, 0mo]; Embeddings [expert, 0mo]
CAREER:
  - Marketing Manager @ TCS (current) [60mo]
  - Marketing Executive @ Infosys [48mo]
{"grade": 0, "fit_score": 0.5, "confidence": "high", "reasoning": "Career is entirely Marketing at services firms; the 'expert' AI skills all show 0 months duration and no assessment — classic keyword-stuffing trap with no ML substance.", "concerns": "non-technical career, stuffed skills"}"""

_FEWSHOT_STAR = """Staff Machine Learning Engineer @ Paytm | 7.0 yrs exp
SUMMARY: Own the candidate-JD matching pipeline; rebuilt hybrid BM25+dense retrieval, lifted NDCG@10 from 0.72 to 0.91.
SKILLS: Semantic Search [expert, 40mo (assessed 0.91)]; Information Retrieval [expert, 38mo]; Qdrant [advanced, 22mo]; Embeddings [advanced, 30mo]
CAREER:
  - Staff ML Engineer @ Paytm (current) [40mo] :: production ranking + retrieval serving 1M+ queries/day
  - Senior ML Engineer @ Swiggy [36mo] :: recommendation ranking
{"grade": 4, "fit_score": 9.6, "confidence": "high", "reasoning": "Senior ML engineer at product companies who literally owns the retrieval/ranking pipeline with a measurable impact metric; skills are backed by years of duration and assessments.", "concerns": "none"}"""


def judge_messages(candidate: dict) -> list[dict]:
    """Build the chat messages to grade one candidate."""
    brief = candidate_brief(candidate)
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "system", "content": f"THE ROLE (ideal candidate):\n{IDEAL_CANDIDATE}"},
        {"role": "user", "content": f"Here is an example TRAP candidate:\n{_FEWSHOT_TRAP}\n\n"
                                    f"Here is an example STAR candidate:\n{_FEWSHOT_STAR}\n\n"
                                    f"Now grade THIS candidate. Return JSON only.\n\n"
                                    f"CANDIDATE:\n{brief}"},
    ]
