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
   │     skills (trust-wtd)    proficiency × duration × assessment cross-check
   │     career quality        product vs services, progression, tenure
   │     experience            5–9 yr soft trapezoid
   │     education             institution tier × field relevance
   │     location              India preference (soft)
   │     semantic              dense embedding of profile prose vs an "ideal
   │                           candidate" query  (src/semantic_matcher.py)
   │
   ├─ Availability multiplier  recency, response rate, notice, verifications …
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
  in ~20–25 s on the full 100K pool.
- All neural work is **precomputed offline** (`precompute/build_embeddings.py`)
  and cached as `cache/embeddings.npz`. The ranker only loads vectors and does a
  numpy dot product. If the cache is absent, the ranker transparently falls back
  to rule-only weights.

---

## Setup

### A. Run the ranker (the submission path — light, any Python ≥3.10)
```bash
pip install numpy
python rank.py --candidates data/candidates.jsonl --out output/submission.csv
python validate_submission.py output/submission.csv     # official format check
```

### B. Rebuild the embedding cache (offline precompute — needs torch)
Done once; not part of the 5-minute budget. Isolated in its own venv so the
submission path stays torch-free:
```bash
uv venv --python 3.12 .venv-embed
uv pip install --python .venv-embed torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-embed sentence-transformers numpy
.venv-embed/bin/python precompute/build_embeddings.py    # writes cache/embeddings.npz
```

---

## Offline evaluation (our replacement for the hidden leaderboard)

There is no live leaderboard, so we validate against a hand-labeled golden set
and the same metrics the challenge scores on (NDCG@10/@50, MAP, P@10).

```bash
python eval/build_golden_set.py     # samples ~55 diverse candidates to label
# fill the 'grade' column (0-4) in eval/golden_labels.csv
python eval/evaluate.py             # composite metrics + model/judgment disagreements
python tests/test_metrics.py        # unit-tests the metric implementations
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
  ranker.py                 Ensemble: fit × availability → ranking
  reasoning.py              Grounded, honest per-candidate reasoning
  scorers/                  title, skills, career, experience, education,
                            location, behavioral
precompute/
  build_embeddings.py       Offline embedding precompute (torch)
  analyze_dataset.py        EDA / honeypot profiling
eval/
  metrics.py                NDCG@k, MAP, P@k
  build_golden_set.py       Build the labeling sheet
  evaluate.py               Score full pool vs golden set
sandbox/app.py              Optional Streamlit explorer over the submission
docs/approach.md            Full methodology write-up
data/job_description.txt    Extracted JD text
data/  cache/  output/  tests/
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

```bash
python rank.py --candidates data/candidates.jsonl --out output/submission.csv
```
(With `cache/embeddings.npz` present this runs the hybrid ranker; without it,
rule-only. Rebuild the cache via Setup B.)
