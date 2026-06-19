# Loop 2 Plan — Role-Aware Expert Mixture

**Date started:** 19 June 2026  
**Current loop number:** 2  
**Submission deadline:** 2 July 2026, 23:59 IST  
**Status:** Complete

---

## Goal of this loop

Decompose the single monolithic "fit" score into multiple specialized expert dimensions, then audit whether the current ranking misses candidates who are exceptional on specific dimensions.

This is primarily an **audit / selector** — not a new ranker.

---

## Results

- Built 7-dimensional expert scores (`src/expert_scores.py`) for all 83K candidates.
- Audit uses final order (ensemble + LLM rerank + expansion), not base ensemble only.
- Hidden gems (ranks 101-500, top-50 on a dimension): **109**
- Multi-dimension gems (3+ dims): **1** (CAND_0065878, rank 127)
- Two-dimension near-misses: 7 (including CAND_0064904 @ rank 102, LinkedIn)
- Top-100 dimension dominance: anti_gaming 74%, nlp_llm 66%, recsys 64%; seniority only 14%
- Fixed tail dedupe bug in `apply_expansion()` — promoted IDs no longer duplicated in tail; top-100 unchanged (verified 0 diff)

## Yellow flags / watchlist

- **CAND_0065878** (rank 127): only 3+ dimension gem — Niramai/Uber/Glance, production_ml 0.805, anti_gaming 0.937. Watchlist for wider expansion.
- **CAND_0064904** (rank 102): LinkedIn AI Engineer, 2 dims; beat CAND_0047721 in Loop 1 pairwise audit. Watchlist only.

## Metrics

- Final ranking changed? **NO**
- Top-20 changed? **NO**
- Hand qrel: unchanged (audit-only)

## Decision

**AUDIT-ONLY / discovery-only. No ranking change.**

Expert mixture is useful for hidden-gem watchlisting and future expansion selection, but not adopted as a ranker. Pi-agent approved: do not grade/promote watchlist candidates in this loop.

Tail dedupe fix in `apply_expansion()` committed (correctness; top-100 unchanged).

## Next recommended loop

**Loop 3: Adversarial Resume / Prompt-Injection Robustness** (pi recommendation)
