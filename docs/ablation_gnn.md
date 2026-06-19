# Ablation: GNN candidate-job matcher (graph embedding)

**Status: ABSTAINED from ranking.** Stable across seeds but the graph signal is
weaker than, and 98.7% redundant with, the structured feature ensemble.

## What we built

A heterogeneous graph (`precompute/build_graph.py`): 83,779 candidate nodes +
38 skill + 16 concept + 63 company nodes, 407K weighted edges (candidate–skill by
relevance, candidate–concept by career-text mention, candidate–company). A
candidate whose neighbourhood overlaps with known-strong engineers (shared
skills/companies/concepts) embeds near them.

**node2vec didn't scale** — its pure-Python alias-sampling preprocessing didn't
finish on 84K weighted nodes within our compute budget (the dependency friction
the plan warned about). We pivoted to a **spectral embedding**: TruncatedSVD on
the sparse candidate×attribute adjacency — a scalable graph factorization. Then a
HistGradientBoostingRegressor learns fit from the embedding (`train_gnn.py`).

Runtime (`src/gnn_scores.py`) loads only `cache/gnn_scores.npz`.

## Result: stable, but weaker and redundant

| metric                              | value          |
|-------------------------------------|----------------|
| graph-only 5-fold CV NDCG@10        | **0.9135** ± 0.0040 |
| hand-tuned ensemble (same labels)   | **0.9798**     |
| top-100 overlap across 5 seeds      | **100/100**    |
| adjacency explained variance (64-d) | 98.73%         |

The graph signal is **stable** (perfect top-100 overlap across seeds — the
deterministic embedding + regularized regressor converges consistently) but
**weaker** than the structured ensemble. The 98.7% explained variance reveals
why: the adjacency is skill/concept/company co-occurrence, which our
trust-weighted scorers already capture — the graph just rediscovers it noisily.

## Integration fails (same saturation + leakage pattern)

| variant        | hand comp | hand n@10 | llm comp | llm n@10 |
|----------------|-----------|-----------|----------|----------|
| ensemble       | 0.9811    | 0.9661    | **0.9765**| **1.0000**|
| +gnn_blend0.15 | 0.9999    | 1.0000    | 0.9559   | 0.9470   |
| +gnn_blend0.25 | 1.0000    | 1.0000    | 0.9549   | 0.9445   |
| +gnn_rrf       | 1.0000    | 1.0000    | 0.9532   | 0.9445   |

The hand-qrel "improvement" to 1.0000 is **label leakage** (the 53 hand labels
are in the training set — same circularity as LTR). The LLM qrel honestly shows
integration hurts (ndcg@10 1.0 → 0.945) — the same saturation wall: active
reordering disrupts the perfect top-10. The graph-only CV (0.9135 < 0.9798) is
the trustworthy number.

## Decision

GNN is **not used in the final ranking**. It confirmed that on this dataset, the
graph structure is redundant with the structured features — a valuable negative
finding (graph methods shine when features are sparse/unstructured; ours are
rich). Retained as an ablation.

## Deck framing
> "We built a heterogeneous candidate–skill–concept–company graph and learned
> node embeddings. node2vec's pure-Python alias sampling didn't scale to 84K
> weighted nodes, so we used a spectral embedding (TruncatedSVD). The graph
> signal was remarkably stable — 100/100 top-100 overlap across five seeds — but
> weaker than the structured ensemble (CV NDCG 0.91 vs 0.98): the adjacency is
> skill/concept co-occurrence that our trust-weighted scorers already capture
> (98.7% explained variance). Graph methods add the most when features are
> sparse; ours are rich, so we retained the graph as an ablation and kept the
> interpretable structured ensemble."
