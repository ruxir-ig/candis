# Evidence-guided LLM Expansion Audit

## Purpose

Find strong high-evidence candidates buried outside the existing top-200 LLM
re-rank window, without disturbing the saturated top-20.

## Selection

From ranks 201-800 (fit >= 0.45, excluding "available but mediocre"), 210
candidates selected across 6 diverse buckets (evidence_graph, semantic, raw fit,
retrieval/vector skills, Search/RecSys/NLP titles, career/product). All graded
by the same model (llama-3.3-70b) as the base cache for score comparability.

Grade distribution: **4: 39, 3: 132, 2: 39** — healthy spread, real discrimination.

## Safety policy (pi's two-gate spec)

- ranks 1-20 **FROZEN**
- elite band (21-50): fit >= rank50_thr + 0.20 (9.40), grade 4, evidence >= 0.85,
  confidence high, no concerns
- normal band (51-100): fit >= rank100_thr + 0.15 (9.35), grade 4, evidence >= 0.75,
  confidence medium/high, no major concerns
- every promotion displaces the weakest incumbent in its band

## Results

- selected: 210  · graded: 210  · eligible: **7** (all elite)  · promoted: **7**

## Entrants (all promoted into ranks 46-52)

| new rank | candidate | LLM fit | title @ company | yrs | audit |
|---:|---|---:|---|---:|---|
| 46 | CAND_0099401 | 9.8 | NLP Engineer @ Dream11 | 7.7 | BM25/Embeddings/Qdrant ✓ |
| 47 | CAND_0030827 | 9.8 | Senior Data Scientist @ Freshworks | 5.4 | NLP/RecSys/FAISS ✓ |
| 48 | CAND_0047721 | 9.8 | Senior Data Scientist @ Microsoft | 7.0 | LTR/Milvus/BM25/RAG, asmt IR=87.8 ✓ |
| 49 | CAND_0094759 | 9.8 | Lead AI Engineer @ Meta | 8.6 | LTR/Semantic Search/Qdrant ✓ |
| 50 | CAND_0060072 | 9.8 | Staff ML Engineer @ Amazon | 5.7 | Sentence Transformers/Milvus ✓ |
| 51 | CAND_0007411 | 9.8 | Senior ML Engineer @ Amazon | 8.0 | OpenSearch/Vector Search, asmt VS=91.3 ✓ |
| 52 | CAND_0092278 | 9.8 | Senior NLP Engineer @ Microsoft | 6.8 | HF Transformers/QLoRA/pgvector ✓ |

All: grade 4, confidence high, **zero concerns**. All at top product companies.

## Leavers (displaced from ranks 94-100)

| old rank | candidate | title @ company |
|---:|---|---|
| 94 | CAND_0007460 | AI Engineer @ Salesforce |
| 95 | CAND_0064904 | AI Engineer @ LinkedIn |
| 96 | CAND_0045250 | Applied ML Engineer @ Rephrase.ai |
| 97 | CAND_0016163 | Applied ML Engineer @ Dream11 |
| 98 | CAND_0018722 | RecSys Engineer @ Saarthi.ai |
| 99 | CAND_0026532 | RecSys Engineer @ Zomato |
| 100 | CAND_0054394 | RecSys Engineer @ PharmEasy |

These were the weakest incumbents (bottom 7 of the top-100). Three appear in the
LLM qrel but were already low-ranked (94-100), so their displacement is immaterial
to NDCG.

## Decision-gate checklist

| gate | status |
|------|--------|
| no changes in ranks 1-20 | ✓ frozen |
| at most 5-15 candidates enter top-100 | ✓ 7 |
| every entrant manually audited | ✓ all clean, no red flags |
| hand-qrel NDCG@10 unchanged | ✓ 0.9307 = 0.9307 |
| hand-qrel NDCG@50 not materially worse | ✓ 0.9799 = 0.9799 |
| top-100 audit clean | ✓ no stuffers/honeypots/services-only |

## Metric impact

| variant | hand comp | hand n@10 | llm comp | llm n@10 |
|---------|-----------|-----------|----------|----------|
| baseline (no expansion) | 0.9563 | 0.9307 | 0.9930 | 1.0000 |
| + expansion | 0.9563 | 0.9307 | 0.9929 | 1.0000 |

NDCG is unchanged (entrants aren't in the qrel; displaced were low-ranked). The
improvement is **qualitative**: 7 genuinely stronger candidates (Microsoft, Amazon,
Meta — the exact companies the JD targets) replace 7 weaker incumbents at the
bottom of the top-100. This is the kind of recall gain the saturated metric
cannot measure but the hidden evaluation rewards.

## Final decision: **ADOPTED**

`rank.py --use-expansion` generates the expansion submission. The base cache
(cache/llm_rerank.jsonl) is untouched; the expansion cache
(cache/llm_expansion.jsonl) is separate for instant revert.
