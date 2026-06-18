# Ablation: BM25 sparse retrieval + Reciprocal Rank Fusion

**Status: ABSTAINED from final** (ablation-only). Kept for the deck as evidence of
an evidence-based selection.

## What we built

A from-scratch Okapi BM25 (`src/sparse_matcher.py`, no `rank-bm25` dependency)
over candidate text, scored against the same distilled ideal-candidate query the
dense bi-encoder uses — the textbook hybrid-retrieval setup (dense + sparse on
one intent). Scores precomputed offline (`precompute/build_sparse.py` →
`cache/sparse_scores.npz`); runtime only loads the cache.

An anti-gaming contradiction gate dampens BM25 where it is high but the
*structured* title and career scores are low (keyword-stuffer fix). After our
coarse filter, **0 candidates tripped the gate** — the worst stuffers are already
removed, so the gate is a pure safety net.

Fusion via Reciprocal Rank Fusion (`src/fusion.py`): `rrf(c) = Σ 1/(60+rank_i(c))`
over the ensemble ranking (which bakes in the availability multiplier) and the
BM25 ranking.

## Result (measured on both qrels, ablation_fusion.py)

| variant            | hand comp | hand ndcg@10 | llm comp | llm ndcg@10 | llm p@10 |
|--------------------|-----------|--------------|----------|-------------|----------|
| ensemble (base)    | **0.9811**| **0.9661**   | **0.9765**| **1.0000** | **1.0000**|
| +bm25 (weighted)   | 0.9547    | 0.9191       | 0.9504   | 0.9470      | 0.9000   |
| +bm25 (RRF, w=1.0) | 0.9562    | 0.9212       | 0.9504   | 0.9470      | 0.9000   |
| +bm25 (RRF, w=0.3) | 0.9581    | 0.9238       | 0.9477   | 0.9470      | 0.9000   |

**BM25 hurts at every weight, on both qrels.** NDCG@10 drops and P@10 falls from
1.0 to 0.9 on the LLM qrel.

## Why it failed (honest diagnosis)

1. **Redundant-but-noisier signal.** Our dense bi-encoder *and* the structured
   title enum ("Search Engineer", "Recommendation Systems Engineer",
   "Staff ML Engineer") already capture retrieval/recommendation relevance — and
   more precisely than lexical term overlap. BM25 adds a correlated, noisier vote
   that can only dilute.
2. **Disrupts the availability-weighted ordering.** The ensemble score encodes
   `fit × (floor + (1-floor)×availability)`. RRF replaces that continuous score
   with a rank-only fusion, throwing away the behavioral multiplier's fine
   gradient — so reachable strong candidates and unreachable strong candidates
   get treated more alike.
3. **Doesn't compose with the LLM re-rank.** RRF shifts the top-200 enough that
   >10% of the re-rank window loses its cached LLM grade, tripping the
   `MIN_COVERAGE=0.9` guard — so `+bm25_rrf +rerank` collapses to `+bm25_rrf`
   (the re-rank silently falls back to the RRF order).

## The one positive finding

RRF *did* surface genuinely buried grade-4 candidates (Recommendation Systems
Engineers at Verloop/Zomato/Zoho, Search Engineers at Rephrase) — exactly the
"buried stars" the recall layer misses. But it pushed out equally-strong
candidates for a net-negative result. This validates the recall gap the later
approaches (evidence graph, LTR) should target more surgically.

## Decision

Per the acceptance gate (NDCG@10 must not drop), BM25 is **not used in the final
ranking**. The generic RRF engine (`src/fusion.py`) is retained for future
rankers that may earn a place. Rank.py defaults to the proven ensemble; BM25
fusion is opt-in via `--rrf` for experimentation.

### Deck framing
> "We added Okapi BM25 sparse retrieval and fused it with the dense ensemble via
> Reciprocal Rank Fusion. Even at weak weights it slightly hurt NDCG@10 because
> our dense semantic model plus the structured title enum already capture lexical
> relevance more precisely than raw term overlap — and RRF discarded the
> availability multiplier's fine gradient. We kept BM25 as an ablation and
> retained the RRF engine for later rankers, but did not use it in the final
> system."
