# Intelligent Candidate Discovery & Ranking

A capability-matching pipeline for the Redrob *Intelligent Candidate Discovery &
Ranking Challenge*: rank the top 100 candidates for a Senior AI/ML Engineer role
from a pool of 100,000 — **without** falling for keyword stuffing.

The core thesis (straight from the JD): *the right answer is not "whoever lists
the most AI keywords."* So the system reasons about the gap between what a
profile **says** and what it **means** — weighting real career history, the
prose of what someone actually built, and behavioral availability over a raw
skill-list match, while detecting logically impossible "honeypot" profiles.

---

## How it works

A multi-stage pipeline. Each candidate gets a **fit** score (how well they match
the role) which is then **dampened by availability** (a great-on-paper candidate
who never answers recruiters isn't really hireable). Availability is a
multiplier, never additive — it can't rescue a bad fit.

```
100K candidates
   │
   ├─ Honeypot filter         logically impossible profiles removed (src/honeypot_detector.py)
   │
   ├─ Coarse filter           HARD safety net: fully non-technical careers rejected
   │                          (src/coarse_filter.py)
   │
   ├─ Fit scoring  ──────────  weighted blend of:
   │     title/role           rule-based, structured fields (src/scorers/)
   │     skills (trust-wtd)   proficiency × duration × assessment cross-check
   │     career quality       product vs services, progression, tenure
   │     experience           5–9 yr soft trapezoid
   │     education            institution tier × field relevance
   │     location             India preference (soft)
   │     semantic             dense embedding of profile prose vs an "ideal
   │                          candidate" query  (src/semantic_matcher.py)
   │
   ├─ Availability multiplier  recency, response rate, notice, verifications …
   │
   ├─ LLM re-rank (cached)     pointwise cross-encoder judgment on the top-200
   │                          (src/llm_reranker.py; grades precomputed offline)
   │
   └─ Rank → top 100 + grounded reasoning → output/submission.csv
```

Why hybrid: the rule scorers read **structured** fields (title enum, skill list);
the semantic layer reads the **prose** (summary + what they did each role), where
genuine fit shows up ("rebuilt candidate-JD matching, 0.72→0.91 NDCG@10") and
where stuffers expose themselves (an HR manager's summary is about HR no matter
what skills they pasted on).

---

## Compute-constraint compliance

The challenge ranking step must run **≤5 min, ≤16 GB, CPU-only, no network**.

- **`rank.py` uses only the Python stdlib + numpy.** No torch, no network. Runs
  in ~13 s on the full 100K pool.
- All neural/LLM work is **precomputed offline** and cached: the MiniLM
  embedding cache (`precompute/build_embeddings.py`), the LLM judge grades
  (`precompute/llm_label.py`), and the experimental ranker caches (BM25, LTR,
  GNN). The ranker only loads caches and does numpy ops. If a cache is absent it
  transparently falls back (no embeddings → rule-only; no grades → no re-rank).

---

## The ranking laboratory

Five research-inspired rankers were tested as *candidates* — each had to beat the
proven ensemble on the non-circular hand qrel without breaking the top-10. One
earned a place (LLM re-rank); the rest are ablations. Full write-ups in `docs/`.

| ranker | outcome | why |
|--------|---------|-----|
| BM25 + RRF | rejected | lexical noise; semantic+title already beat it |
| Evidence graph | reasoning enrichment | corroboration scoring; ranking breaks the saturated top-10 |
| LTR calibrator | rejected | overfits 219 labels (CV 0.90 < 0.94); Ridge validates weights |
| GNN graph embedding | rejected | stable but redundant with rich features |
| **LLM re-rank** | **adopted** | cross-encoder judgment; improves qualitative membership |

**Anti-gaming robustness** (`eval/robustness_audit.py`): injecting 7 expert AI
skills into 500 weak profiles puts **15 in the top-100 under keyword matching,
0 under our system.** The deck's hero slide.

---

## Setup

### A. Run the ranker (the submission path — uv-managed, numpy-only)
```bash
uv run python rank.py --candidates data/candidates.jsonl --out output/submission.csv
uv run python validate_submission.py output/submission.csv     # official format check
```

### B. Rebuild the offline caches (not part of the 5-minute budget)
```bash
# Embeddings (separate torch venv)
uv venv --python 3.12 .venv-embed
uv pip install --python .venv-embed torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-embed sentence-transformers numpy
.venv-embed/bin/python precompute/build_embeddings.py          # cache/embeddings.npz

# Experimental ranker caches (optional, for the ablations)
uv run --extra deck python precompute/build_sparse.py          # BM25
uv run --extra ltr python precompute/train_ltr.py              # LTR
uv run --extra graph python precompute/build_graph.py          # GNN graph
uv run --extra graph python precompute/train_gnn.py            # GNN scores
```

### C. Build the deck
```bash
uv run --extra deck python precompute/build_deck.py            # output/deck.pptx + deck.pdf
```

---

## Offline evaluation (our replacement for the hidden leaderboard)

There is no live leaderboard, so we validate against a hand-labeled golden set
and the same metrics the challenge scores on (NDCG@10/@50, MAP, P@10).

```bash
uv run python eval/evaluate.py             # composite metrics (auto qrel) + disagreements
uv run python eval/evaluate.py --qrel hand # non-circular hand labels (53)
uv run python tests/test_metrics.py        # unit-tests the metric implementations
uv run python eval/ablation_report.py      # consolidated ablation table
uv run python eval/robustness_audit.py     # anti-gaming audit → docs/robustness_audit.md
```

---

## Repository layout

```
rank.py                     Single entry point → submission.csv
src/
  config.py                 All weights & thresholds (tune here)
  loader.py                 Streaming JSONL/gzip loader
  honeypot_detector.py      Stage 1: logical-impossibility detection
  coarse_filter.py          Stage 2: hard non-technical-career safety net
  profile_text.py           Candidate → text for embedding
  jd_query.py               Distilled "ideal candidate" query (not the raw JD)
  semantic_matcher.py       Cached-embedding cosine scoring (numpy only)
  evidence_graph.py         7-category cross-field corroboration scorer
  llm_judge.py / llm_client.py / candidate_brief.py   offline LLM judge
  llm_reranker.py           Cached pointwise LLM re-rank of top-200
  ranker.py                 Ensemble: fit × availability → ranking
  reasoning.py              Grounded, honest per-candidate reasoning (+ evidence)
  fusion.py                 Reciprocal Rank Fusion (for experiments)
  scorers/                  title, skills, career, experience, education,
                            location, behavioral
precompute/
  build_embeddings.py       Offline embedding precompute (torch, .venv-embed)
  build_sparse.py           BM25 sparse score cache
  train_ltr.py              LTR calibrator (sklearn) → ltr_scores.npz
  build_graph.py / train_gnn.py   GNN graph + spectral embedding
  llm_label.py / llm_probe.py     offline LLM judge batch labeler
  build_deck.py             Submission deck (PPTX + PDF)
eval/
  metrics.py                NDCG@k, MAP, P@k (+ unit tests)
  evaluate.py               Score full pool vs golden/LLM qrels
  ablation_report.py        Consolidated ablation table
  ablation_fusion.py / ablation_evidence.py / ablation_ltr.py / ablation_gnn.py
  robustness_audit.py       Anti-gaming perturbation audit
sandbox/app.py              Optional Streamlit explorer over the submission
docs/                       ablation write-ups + robustness audit + final report
output/submission.csv       the ranked top-100
output/deck.pdf             the submission deck
data/  cache/  tests/
```

---

## Optional: interactive sandbox

```bash
pip install -r requirements-sandbox.txt
streamlit run sandbox/app.py
```

A read-only explorer over the top-100: leaderboard, per-candidate score
breakdown (component bar chart), skills, behavioral signals, career history,
and a honeypot/trap audit view.

---

## Reproduce the submission in one command

By default, the final ranker also applies a cached **evidence-guided LLM expansion**
pass: it grades a diverse, evidence-rich slice of ranks 201-800 with the same
judge, then promotes only manually-audited, high-confidence grade-4 candidates
into ranks 21-100 (top-20 frozen). This recovered 7 Senior/Staff ML engineers
at Microsoft, Amazon, and Meta. Disable with `--no-expansion`.

```bash
uv run python rank.py --candidates data/candidates.jsonl --out output/submission.csv
```
(With `cache/embeddings.npz` + `cache/llm_rerank.jsonl` present this runs the full
hybrid + LLM re-rank + expansion pipeline; without caches it degrades gracefully.)
The offline LLM label caches are committed so the re-rank is reproducible without
an API key; the embedding cache is regenerable via Setup B.
