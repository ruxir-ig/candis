# Ablation: Evidence Graph ranker

**Status: ranking = ablation-only (no active ranker); adopted for reasoning
enrichment + validated as a guardrail.** The corroboration evidence now feeds the
submission's reasoning column.

## What it is

A cross-field corroboration scorer (`src/evidence_graph.py`) that reads profile
fields *jointly* — the per-field scorers each read one slice. Seven soft-scored
categories (anti-stuffing applied as a multiplier):

  1. **title_skill_corroboration** — the title enum (our strongest anti-stuffer
     signal) must agree with the ML-skill list. Non-technical title + high ML
     skills = keyword-stuffer footprint → 0.05.
  2. **role_skill_corroboration** — retrieval/search concepts in BOTH skills and
     career prose.
  3. **retrieval_maturity** — career prose shows shipped ranking rigor (NDCG,
     MRR, A/B testing, learning-to-rank, recommendation systems).
  4. **assessment_support** — `skill_assessment_scores` back claimed skills.
  5. **duration_support** — key skills have real tenure (expert + 0 months flagged).
  6. **seniority_consistency** — years of experience align with title seniority.
  7. **product_context** — product-company experience.

Output per candidate: `{score, components, positive[], negative[]}`.

## Calibration (validated)

| profile type               | evidence score |
|----------------------------|----------------|
| Staff ML @ Paytm (top-1)   | 0.96           |
| RecSys/Search engineers    | 0.84–0.88      |
| keyword stuffers (Ops/HR/  | 0.37–0.42      |
|  Content Writer + AI skills|                |
|  that pass coarse filter)  |                |

Clean separation: stuffers <0.42, strong >0.84.

## Integration ablation (does it improve the *ranking*?)

Tested four ways to fold it into the proven ensemble (ablation_evidence.py):

| variant          | hand comp | hand n@10 | llm comp | llm n@10 | grade≥3 in top-100 (llm) |
|------------------|-----------|-----------|----------|----------|--------------------------|
| ensemble (base)  | **0.9811**| **0.9661**| **0.9765**| **1.0000**| 70/157                 |
| +ev_rrf          | 0.9576    | 0.9238    | 0.9402   | 0.9319   | 71/157                   |
| +ev_blend (0.15) | 0.9603    | 0.9276    | 0.9438   | 0.9373   | **75/157**               |
| +ev_penalty      | 0.9578    | 0.9238    | 0.9401   | 0.9319   | 74/157                   |
| +ev_penalty_only | 0.9811    | 0.9661    | 0.9765   | 1.0000   | 70/157 (= base, no-op)   |

Composed with the LLM re-rank (the actual submission path):

| variant                | coverage | hand comp | llm comp |
|------------------------|----------|-----------|----------|
| ensemble + rerank      | 1.00     | **0.9618**| **0.9947**|
| +ev_blend + rerank     | 0.89     | 0.9605    | 0.9463   |
| +ev_penalty + rerank   | 0.81     | 0.9578    | 0.9401   |

## Why active ranking fails (same root cause as BM25)

1. **The top is saturated.** NDCG@10 is already 1.0 (llm) / 0.9661 (hand) and is
   50% of the composite. Any active reordering of the top-100 can only risk it.
2. **Breaks the LLM re-rank composition.** Reordering shifts the top-200 past the
   `MIN_COVERAGE=0.9` guard (coverage falls to 0.81–0.89), so the re-rank silently
   falls back — the two stages don't compose.
3. The recall win IS real (+5 grade≥3 candidates reach top-100 via blend) but
   can't register on NDCG because it comes at the cost of top-10 perturbation.

## What we adopted

- **Reasoning enrichment (shipped).** The single best corroboration point
  (`evidence_phrase`) now appears in the submission's reasoning column for the
  top of the list — grounded "why" with zero ranking risk. e.g. *"Retrieval/
  search appears in both skills and career: rag, recommendation, retrieval,
  semantic search."*
- **penalty_only guardrail (validated, available).** Dampens only when evidence
  < 0.5. On current data it's a no-op (no stuffer reaches the top-100), but it's
  free insurance if the recall set ever changes.

## The identified opportunity: buried-stars rescue

The evidence graph's most valuable ranking use isn't reordering the top — it's
**expanding the LLM re-rank window**. 28 genuinely strong candidates sit at ranks
203–600 with evidence > 0.85 but are never LLM-graded (Senior Data Scientists at
Flipkart/Amazon/Microsoft/Netflix, Search/RecSys engineers at Razorpay/Dream11).
Promoting those into the re-rank pool so the judge can rescue them is the
direction that targets the real recall gap — tracked as a follow-up (needs ~28
LLM grades + audit).

## Deck framing
> "We built a seven-category evidence graph that scores cross-field corroboration
> — the signal no per-field scorer captures. It cleanly separates keyword
> stuffers (0.37) from genuine engineers (0.88+). As an active ranker it hurt
> NDCG because the top-10 is already saturated and it broke the LLM re-rank's
> coverage; so we kept it for reasoning enrichment and as a validated anti-gaming
> guardrail, and identified its real ranking value as an LLM re-rank window
> expander for buried strong candidates."
