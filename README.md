# Candis — Intelligent Candidate Discovery & Ranking

A capability-matching pipeline for the Redrob *Intelligent Candidate Discovery & Ranking Challenge*: rank the top 100 candidates for a Senior AI/ML Engineer role from 100,000 profiles — **without** rewarding keyword stuffing.

The core thesis (straight from the JD): *the right answer is not "whoever lists the most AI keywords."* Candis reasons about the gap between what a profile **says** and what it **means** — weighting real career history, the prose of what someone built, and behavioral availability over raw skill-list match, while detecting logically impossible "honeypot" profiles.

---

## How it works

A multi-stage pipeline. Each candidate gets a **fit** score (how well they match the role) dampened by **availability** (a multiplier, never additive — it can't rescue a bad fit).

```
100K candidates → honeypot filter → coarse filter → fit scoring → availability
multiplier → LLM re-rank (cached) → evidence-guided expansion → top-100 + reasoning
```

The rule scorers read **structured** fields (title, skill list); the semantic layer reads the **prose** (summary + role descriptions), where genuine expertise shows up ("rebuilt candidate-JD matching, 0.72→0.91 NDCG@10") and stuffers expose themselves (an HR manager's summary is about HR no matter what skills they pasted).

**Anti-gaming robustness**: injecting 7 expert AI skills into 500 weak profiles puts **15 in the top-100 under keyword matching, 0 under Candis.**

---

## Compute-constraint compliance

Ranking must run **≤5 min, ≤16 GB, CPU-only, no network**.

- `rank.py` uses stdlib + numpy only. Runs in ~13 s on the full 100K pool.
- All neural/LLM work is **precomputed offline and cached**. Missing caches trigger graceful fallback (no embeddings → rule-only; no grades → no re-rank).

---

## Reproduce the submission

```bash
uv run python rank.py --candidates data/candidates.jsonl --out output/submission.csv
uv run python rank.py --candidates data/candidates.jsonl --out output/submission.xlsx
```

Use CSV for local validation and XLSX for the challenge upload form. Installation: Python 3.12 + [`uv`](https://docs.astral.sh/uv/). All LLM grades are committed; no API key needed at ranking time.

---

## Repository layout

```
rank.py                     Entry point → submission.csv
src/
  config.py                 All weights & thresholds
  ranker.py                 Ensemble: fit × availability → ranking
  llm_reranker.py           Cached pointwise LLM re-rank of top-200
  llm_expansion.py          Cached evidence-guided promotion pass
  semantic_matcher.py       Cached-embedding cosine scoring (numpy only)
  evidence_graph.py         7-category cross-field corroboration scorer
  reasoning.py              Grounded, honest per-candidate reasoning
  scorers/                  title, skills, career, experience, education,
                            location, behavioral scorers
  honeypot_detector.py      Stage 1: logical-impossibility detection
  coarse_filter.py          Stage 2: hard non-technical-career safety net
precompute/                 Offline cache builders for embeddings and LLM labels
eval/                       Metrics, evaluation, robustness and top-100 audits
docs/                       Final approach, robustness, and top-100 audit notes
output/                     Generated CSV/XLSX artifacts; kept out of git except .gitkeep
```

---

## Final validation

- **6 research loops**: pairwise LLM audit (83.7% entrant win rate), expert-mixture hidden-gem scan, adversarial prompt-injection (0 affect final ranking), rank stability test, fairness audit, expansion v2 (buried-star pool exhausted). All audit-only, zero ranking changes.
- **Hand qrel**: NDCG@10 = 0.93+ across all pipeline modes (labels are polarized/saturated).
- **Top-100 audit**: zero red flags — 0 honeypots, 0 non-technical, 0 services-only, 0 stuffers.
