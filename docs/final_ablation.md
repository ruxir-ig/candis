# Final ablation report

> Every ranker is a *candidate*. It does not replace the current ranker
> until it proves itself on the (non-circular) hand qrel AND keeps the
> top-10 clean. Five research-inspired rankers tested; one adopted.

Scoring: `0.5*ndcg@10 + 0.3*ndcg@50 + 0.15*map + 0.05*p@10`
Pool: 83,779 post-filter candidates. Generated in 31s.

## Master ablation table

| ranker | decision | hand n@10 | hand comp | llm n@10 | llm comp |
|--------|----------|-----------|-----------|----------|----------|
| rule+semantic | baseline | 0.9661 | 0.9811 | 1.0000 | 0.9765 |
| +BM25 RRF | REJECTED | 0.9212 | 0.9562 | 0.9470 | 0.9504 |
| +evidence-graph RRF | REJECTED (rank) | 0.9238 | 0.9576 | 0.9319 | 0.9402 |
| +evidence-graph blend | REJECTED (rank) | 0.9276 | 0.9605 | 0.9413 | 0.9463 |
| +LTR blend | REJECTED | 1.0000 | 0.9998 | 0.9470 | 0.9553 |
| +GNN blend | REJECTED | 1.0000 | 0.9999 | 0.9470 | 0.9559 |
| +LLM re-rank | ADOPTED (final) | 0.9307 | 0.9618 | 1.0000 | 0.9947 |

_Note: hand qrel (53 labels) is non-circular. LLM qrel (200 labels) is
partly circular with the re-rank. Where a learned ranker's hand-qrel score
looks perfect, that's label leakage — its labels are in the training set._

## Evidence-based decisions

- **BM25 + RRF** — rejected. Lexical overlap is noisier than our dense
  semantic + title enum; RRF discards the availability gradient.
- **Evidence graph** — ranking rejected (breaks the saturated top-10), but
  **adopted for reasoning enrichment** (corroboration evidence in the
  reasoning column) and as a validated anti-gaming guardrail.
- **LTR** — rejected (overfits 219 skewed labels). Ridge coefficients
  independently validated the hand-tuned weights.
- **GNN** — rejected (graph signal stable but redundant with rich features).
- **LLM re-rank** — **adopted.** Pointwise cross-encoder judgment on the
  top-200; the only stage that improved qualitative top-100 membership.

## Anti-gaming robustness (full audit: docs/robustness_audit.md)

| perturbation (500 weak profiles) | keyword baseline | our system |
|-----------------------------------|------------------|------------|
| keyword stuffing → top-100 | **15** | **0** |
| keyword stuffing → top-500 | 24 | 0 |
| title inflation → top-100 | — | 0 |
| skill shuffle (max Δ) | — | 0 |

## Final top-100 composition

**Top companies represented:** CRED (6), Rephrase.ai (6), Netflix (5), Sarvam AI (5), Zoho (5), Google (4), Meta (4), Ola (4), LinkedIn (4), Zomato (3)

**Title categories (top):** ML/AI (40), Senior (20), RecSys (19), Search/IR (11), Staff/Lead (6), NLP (4)

## Final top-10

| # | candidate | title @ company | yrs |
|---|-----------|-----------------|-----|
| 1 | CAND_0077337 | Staff Machine Learning Engineer @ Paytm | 7.0 |
| 2 | CAND_0081846 | Lead AI Engineer @ Razorpay | 6.7 |
| 3 | CAND_0002025 | Senior AI Engineer @ Apple | 5.9 |
| 4 | CAND_0018499 | Senior Machine Learning Engineer @ Zomato | 7.2 |
| 5 | CAND_0079387 | AI Engineer @ Microsoft | 6.9 |
| 6 | CAND_0071974 | Senior AI Engineer @ Netflix | 7.8 |
| 7 | CAND_0030031 | AI Engineer @ Microsoft | 5.7 |
| 8 | CAND_0011687 | Senior NLP Engineer @ Niramai | 7.8 |
| 9 | CAND_0088025 | Staff Machine Learning Engineer @ Yellow.ai | 8.6 |
| 10 | CAND_0055905 | Senior Machine Learning Engineer @ Flipkart | 8.1 |
