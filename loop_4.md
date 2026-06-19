# Loop 4 Plan — Rank Stability Audit

**Date started:** 19 June 2026  
**Current loop number:** 4  
**Submission deadline:** 2 July 2026, 23:59 IST  
**Status:** Complete

---

## Goal

Test whether the final ranking is robust to cache/model dependencies. Answer: "Is the system overdependent on one cache or model?"

## Method

Run `rank.py` in degraded modes and compare top-K overlap with the default submission:

| Mode | Semantic | Rerank | Expansion |
|------|----------|--------|-----------|
| full (default) | ✓ | ✓ | ✓ |
| no_expansion | ✓ | ✓ | ✗ |
| no_rerank | ✓ | ✗ | ✗ |
| base_ensemble | ✓ | ✗ | ✗ |
| rule_only | ✗ | ✗ | ✗ |

For each mode, compute:
- Top-20 overlap with default
- Top-50 overlap
- Top-100 overlap
- Hand qrel NDCG@10, NDCG@50, MAP

## Adoption gates

Audit-only. No ranking change.

## Results

- Top-100 pool is 82% stable without any ML (rule-only retains 82/100 candidates)
- Top-20 ordering is 55% dependent on LLM rerank (9/20 positions change without it)
- Expansion blast radius: exactly 7 seats (all in ranks 46-100, top-20 frozen)
- Hand qrel is saturated: rule-only scores higher on hand qrel (0.9647 vs 0.9307 NDCG@10) because LLM rerank optimizes for the de-polarized LLM qrel instead
- Graceful degradation: rule-only mode still produces valid 100 candidates with NDCG@10=0.965

## Decision

AUDIT-ONLY. No ranking change. System has healthy defense-in-depth: no single layer is a single point of failure.
