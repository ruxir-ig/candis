# Expert Mixture Audit — Role-Aware Dimension Decomposition

**Loop:** 2  
**Branch:** `research/loop-2-expert-mixture`  
**Date:** 19 June 2026  
**Status:** Complete  

---

## Purpose

Decompose the monolithic fit score into 7 recruiter-style expert dimensions and audit whether the **final** ranking (ensemble + LLM rerank + expansion) misses candidates who are exceptional on specific dimensions.

This loop is **audit / selector only** — not a new ranker.

---

## Method

Seven rule-based expert dimensions (0–1 each):

| Dimension | What it measures |
|-----------|------------------|
| `ir_search` | Retrieval, vector DBs, BM25, embeddings |
| `recsys_ranking` | LTR, recommendation, ranking systems |
| `nlp_llm` | NLP, transformers, RAG, fine-tuning |
| `production_ml` | Production ML signals + deployment language |
| `seniority` | Title level, YOE band, product company |
| `availability` | Behavioral hireability (existing scorer) |
| `anti_gaming` | Evidence-graph corroboration score |

Audit uses the **same final order as `rank.py`** (cached LLM rerank + expansion). Hidden gems = ranked 101–500 overall but in the per-dimension top-50.

---

## Results

### Top-100 dimension dominance

How many of each dimension's top-50 sit in the overall top-100?

| Dimension | Overlap | Interpretation |
|-----------|---------|----------------|
| anti_gaming | 37/50 (74%) | Top-100 strongly corroborated |
| nlp_llm | 33/50 (66%) | Good coverage |
| recsys_ranking | 32/50 (64%) | Good coverage |
| production_ml | 15/50 (30%) | Many production specialists outside top-100 |
| ir_search | 11/50 (22%) | IR specialists often buried |
| availability | 10/50 (20%) | Reachable candidates not always top-ranked |
| seniority | 7/50 (14%) | Seniority alone is a weak selector |

**Key insight:** The final ranker already optimizes for **balanced, corroborated fit** (anti-gaming 74%). Single-dimension peaks (seniority, availability) are poor hidden-gem signals — many rank-1000+ profiles score 1.0 on seniority with weak technical fit.

### Hidden gems (ranks 101–500, top-50 on a dimension)

| Metric | Value |
|--------|-------|
| Total hidden gems | 109 |
| Multi-dimension (3+) | **1** |
| Two-dimension | 7 |

### Multi-dimension gem (only credible expansion candidate)

**CAND_0065878** — rank 127  
- Senior Data Scientist @ Niramai, 7.8y  
- Career: Niramai → Uber → Glance → Rephrase.ai (product companies)  
- Strong: production_ml 0.805, anti_gaming 0.937, recsys_ranking 0.319  
- Profile completeness 90.8%, 15-day notice  
- **Assessment:** Credible buried star; natural watchlist for a wider expansion window. Not promoted automatically — no LLM expansion grade in cache, no adoption-gate evidence to swap into top-100.

### Two-dimension near-misses (watchlist)

| ID | Rank | Dimensions | Note |
|----|------|------------|------|
| CAND_0064904 | 102 | nlp_llm, anti_gaming | AI Engineer @ LinkedIn; beat CAND_0047721 in pairwise audit |
| CAND_0027801 | 110 | recsys, nlp | InMobi NLP, strong vector/LTR skills |
| CAND_0094056 | 129 | recsys, nlp | Rephrase.ai / Adobe |
| CAND_0024620 | 136 | recsys, nlp | PharmEasy AI Engineer |
| CAND_0070398 | 193 | recsys, anti_gaming | |
| CAND_0086151 | 212 | recsys, anti_gaming | |
| CAND_0087630 | 214 | nlp, anti_gaming | |

### Mean expert scores by group

| Group | ir_search | recsys | nlp_llm | prod_ml | seniority | availability | anti_gaming |
|-------|-----------|--------|---------|---------|-----------|--------------|-------------|
| top-100 | 0.449 | 0.186 | 0.426 | 0.472 | 0.820 | 0.765 | **0.896** |
| ranks 101-200 | 0.425 | 0.087 | 0.264 | 0.451 | 0.724 | 0.770 | 0.795 |
| ranks 201-500 | 0.347 | 0.052 | 0.188 | 0.479 | 0.688 | 0.704 | 0.735 |
| hidden gems | 0.436 | 0.100 | 0.270 | 0.469 | 0.728 | 0.734 | 0.789 |

Top-100 dominates on anti-gaming and seniority; hidden gems are weaker on average across all dimensions — consistent with the ranker already surfacing balanced profiles.

---

## Bug fix (tail dedupe)

`apply_expansion()` left promoted candidates duplicated in the post-top-100 tail (same ID at rank ~50 and ~400). Fixed by filtering promoted IDs from the tail. **Top-100 submission unchanged** (verified: 0 diff vs prior `output/submission.csv`).

---

## Decision

**AUDIT-ONLY. No ranking change.**

Rationale:
1. Only **1** candidate strong on 3+ dimensions in ranks 101–500 — insufficient signal for a ranker change.
2. Top-100 already leads on anti-gaming (74% dimension dominance) — the system is doing its job.
3. Expert mixture is useful as a **future expansion selector** and deck narrative ("7-expert recruiter panel"), not as a replacement ranker.
4. CAND_0065878 and CAND_0064904 added to watchlist for Loop 3+ expansion widening.
5. No adoption gates triggered; top-20 unchanged; hand metrics unchanged.

---

## Artifacts

| File | Description |
|------|-------------|
| `src/expert_scores.py` | 7-dimensional expert score computation |
| `eval/audit_expert_mixture.py` | Hidden gems + dominance audit (final order) |
| `docs/expert_mixture_audit.md` | This report |
| `src/llm_expansion.py` | Tail dedupe fix (no top-100 change) |
