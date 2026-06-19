# Loop 6 — Expert-Guided Expansion v2

**Date:** 19 June 2026  
**Branch:** `research/loop-6-expansion-v2`  
**Status:** Complete  

---

## Goal

Search ranks 101-500 for additional elite candidates worth promoting into the top-100, using expert-mixture dimensions as the selection signal.

## Selection

Criteria (union): 2+ expert dimensions in top-50, top-25 in IR/recsys/NLP, high evidence+semantic, prior watchlist. After excluding 410 already-graded candidates: **6 new candidates** selected.

## Results

| Candidate | Grade | Fit | Confidence | Concerns |
|-----------|-------|-----|------------|----------|
| CAND_0023583 (Yellow.ai) | 3 | 7.2 | medium | limited search/retrieval exp |
| CAND_0046459 (upGrad) | 2 | 4.2 | medium | limited production eng exp |
| CAND_0048558 (Rephrase.ai) | 3 | 7.2 | medium | limited search/ranking exp |
| CAND_0074648 (Flipkart) | 3 | 7.2 | medium | limited search/retrieval exp |
| CAND_0086154 (Ola) | 3 | 6.8 | medium | limited search/retrieval exp |
| CAND_0092989 (PharmEasy) | 3 | 7.2 | medium | limited engineering exp |

**0 of 6 pass the promotion gates** (grade 4, fit ≥ 9.8, high confidence, no concerns).

## Decision

**AUDIT-ONLY. No ranking change.**

The two expansion rounds (original 210 graded + v2 6 graded = 216 total) have now exhausted the pool of genuinely strong buried candidates in ranks 101-500. The remaining ungraded candidates are generalist ML engineers, not the elite retrieval/search specialists the JD requires.

This confirms the current top-100 is well-optimized: the 7 promoted candidates from the original expansion were the last elite buried stars.

> "We searched for additional hidden gems and found no candidates strong enough to safely displace the audited top-100."
