# Approach: Intelligent Candidate Discovery & Ranking

A methodology write-up for the Redrob challenge: rank the top 100 candidates
for a Senior AI/ML Engineer role out of 100,000, without falling for keyword
stuffing.

---

## 1. The core thesis

The JD is explicit that the dataset is *baited* with a keyword-stuffing trap:

> "The right answer is not 'find candidates whose skills section contains the
> most AI keywords.' That's a trap we've explicitly built into the dataset."

So the whole system is designed around one question: **does the profile's
substance match its surface?** A Marketing Manager who pasted "RAG, FAISS,
Pinecone" into their skills list must score low no matter how good the skill
list looks. Conversely, a Search Engineer who "rebuilt the candidate-JD
matching pipeline, 0.72 → 0.91 NDCG@10" is a strong fit even without those
exact keywords.

We attack that gap from two directions:

- **Rule scorers read structured fields** (title enum, trust-weighted skills,
  career pattern, education tier). These are robust to prose gaming and anchor
  the defense against stuffers.
- **A semantic layer reads the prose** (summary + what they did each role),
  where genuine fit reveals itself and where stuffers expose themselves (an HR
  manager's summary is about HR, no matter what skills they pasted on).

---

## 2. Pipeline

```
100K candidates
   │
   ├─ Stage 1  Honeypot filter     logically impossible profiles removed
   │                              (YOE mismatch, expert skills w/ 0 duration)
   │
   ├─ Stage 2  Coarse filter       HARD safety net: any candidate whose ENTIRE
   │                              career is non-technical (Marketing / HR /
   │                              Sales / Accounting / …) is rejected outright
   │
   ├─ Stage 3  Fit scoring         weighted blend of 7 dimensions (each ∈ [0,1]):
   │            ├─ title           structured, first-match role enum
   │            ├─ skills          proficiency × duration, assessment cross-check
   │            ├─ career          product vs services, progression, tenure
   │            ├─ experience      5–9 yr soft trapezoid (decays, never zeros)
   │            ├─ education       institution tier × field relevance
   │            ├─ location        India preference (soft)
   │            └─ semantic        cosine(profile prose, "ideal candidate" query)
   │
   ├─ Stage 4  Availability       behavioral availability ∈ [0,1], applied as a
   │        multiplier              MULTIPLIER on fit — never additive
   │
   └─ Stage 5  Rank + reason       sort → top 100 → grounded reasoning strings
```

**Final score:** `fit × (0.65 + 0.35 × availability)`.

The behavioral floor (0.65) is deliberately high: a moderately-hard-to-reach
candidate (e.g. 120-day notice) is *not* unavailable, and over-dampening
buries genuinely strong people. Availability can only ever *reduce* a fit
score, never rescue a bad one.

---

## 3. Why each scorer looks the way it does

### Title / role fit (highest weight, ~0.28–0.33)
This is the strongest single defense against the trap. A `title_role_score`
walks an ordered, first-match-wins enum: genuine ML/Search/RecSys roles score
1.0, wrong-domain titles (Marketing, HR, Sales, Accountant) score 0.05–0.10
*before* the generic `engineer` catch-all, so a non-technical manager never
inherits generic engineering credit. We weight current title 0.7 + best
historical title 0.3 (someone who *just* moved out of an ML role is still
plausible).

### Skills, trust-weighted
A raw keyword count is exactly what the JD warns against. Instead each skill
contributes `relevance_weight × proficiency × duration`, and we cross-check
against the platform's objective `skill_assessment_scores`. A stuffed skill
(high proficiency, ~0 duration, no assessment) contributes almost nothing.

### Career quality
The JD says people whose entire career is at TCS/Infosys/Wipro/Accenture/
Cognizant/Capgemini are "not a fit." We treat "not in the known services list"
as a weak proxy for product-company, and corroborate with title progression
(senior marker appears in a recent role but not the earliest) and tenure
stability (≥3 short non-current stints reads as title-chasing and is docked).

### Education
Small weight (~0.05 — the JD doesn't emphasize it). Institution tier
(tier_1 = IIT/IISc/Stanford/CMU/BITS/NIT, provided in the dataset) ×
field-of-study relevance (CS/ML/AI/DS strong; Stats/Electronics adjacent),
taking the **best** education entry so a late-career IIT M.Tech beats a
tier-3 undergrad.

### Experience
JD targets 5–9 years but explicitly considers strong candidates outside the
band. So we use a soft trapezoid: full credit in 5–9, ramping up from ~3, and
*decaying* (not zeroing) for very senior people.

### Location
Soft India preference (Pune/Noida/Hyderabad/Mumbai/Delhi NCR/Bangalore), never
a hard reject. Relocation willingness is credited.

### Semantic (Stage 4, hybrid only)
We do **not** embed the raw JD — roughly half of it describes what they *don't*
want ("if your AI experience is just LangChain calling OpenAI…",
"title-chasers", "consulting-only"), and embedding all that pollutes the query
vector with negative concepts. Instead we embed a distilled, positive "ideal
candidate" paragraph phrased the way a strong candidate would describe
themselves, so it lands in the same neighbourhood as the candidate prose we
compare it to. Scores are min-max normalized (1st/99th percentile) to sit on
the same scale as the rule scorers.

---

## 4. Compute-constraint compliance

The ranking step must run **≤5 min, ≤16 GB, CPU-only, no network**.

- **`rank.py` uses only the Python stdlib + numpy.** No torch, no network.
  Runs in ~15s on the full 100K pool.
- All neural work is **precomputed offline** (`precompute/build_embeddings.py`)
  and cached as `cache/embeddings.npz`. The ranker loads vectors and does a
  numpy dot product.
- If the cache is absent, the ranker **transparently falls back to rule-only
  weights** — the submission never hard-depends on torch.

This split is enforced by dependency isolation: heavy ML deps live in a
throwaway `.venv-embed` on Python 3.12; the submission path runs on any
Python ≥3.10.

---

## 5. Honeypot + coarse-filter guarantees

The hard filters are *guarantees*, not soft preferences:

- **Honeypots** (logically impossible profiles): flagged on YOE-vs-career-total
  mismatch (>1.5 yr) and the expert-skill-with-zero-duration paradox. 70 caught.
- **Coarse filter**: a candidate whose *entire* career history is an
  unambiguous non-technical domain (Marketing, Sales, HR, Recruiting,
  Accounting, Content, Design, Customer Support, Teaching, Legal, Medical) is
  rejected outright. 16,151 caught — the bulk of the keyword-stuffer population.

These two stages together mean the soft scorers below them can *never* surface
a trap candidate into the top-100, regardless of how the skills/semantic scores
get gamed. The final top-100 audit verifies the top-100 contains zero of each.

---

## 6. Offline evaluation (our replacement for the hidden leaderboard)

There is no live leaderboard, so we manufacture our own small qrel:

1. A human reads a stratified sample like a recruiter and grades 0–4 into
   `eval/golden_labels.csv`.
2. `eval/evaluate.py` scores the full pool, restricts the ordering to the
   labeled set, and reports NDCG@10/@50, MAP, P@10 and the composite the
   challenge uses (`0.5·NDCG@10 + 0.3·NDCG@50 + 0.15·MAP + 0.05·P@10`).

**Primary evaluation (200 LLM labels, de-polarized qrel):**

This is the more discriminating evaluation set because it contains many grade-2
and grade-3 borderline candidates, not just obvious stars/traps.

| metric | value |
|---|---|
| NDCG@10 | 1.0000 |
| NDCG@50 | 1.0000 |
| MAP | 0.9648 |
| P@10 | 1.0000 |
| **composite** | **0.9947** |

**Hand-label sanity check (53 labels, polarized qrel):**

| metric | value |
|---|---|
| NDCG@10 | 0.9307 |
| NDCG@50 | 0.9880 |
| MAP | 1.0000 |
| P@10 | 1.0000 |
| **composite** | **0.9618** |

**LLM judge probe:** before trusting cached grades, `precompute/llm_probe.py`
was run against the hand labels using `meta/llama-3.3-70b-instruct` via NVIDIA
NIM. Result: 7/9 exact agreement and 8/9 within one grade.

### Honest caveats
- **It's a proxy, not a prediction.** The golden set is small and somewhat
  polarized (clear stars vs clear traps), so the ranker separates them easily.
  The value is *relative* — does it move when we tune, and do disagreements
  resolve.
- **The semantic weight is not data-tuned.** On this golden set the composite
  is saturated (flat at ~0.977 from semantic weight 0.00 through 0.25), so
  semantic's weight (0.15) is a judgment call, not a tuned one. We keep it
  modest on purpose so the trust-weighted skills scorer stays the lead signal
  — semantic alone can be gamed by prose that simply mentions AI a lot.
- The top-10 of the submission are textbook JD matches: Staff/Senior/Lead ML &
  AI Engineers at product companies (Paytm, Apple, Razorpay, Zomato, CRED,
  Netflix, Sarvam AI), 5.7–8.6 yrs, real retrieval/vector-DB skills, high
  response rates, short notice.

---

## 7. Reproduce

```bash
# Submission path (CPU, ~15s, no network):
pip install numpy
python rank.py --candidates data/candidates.jsonl --out output/submission.csv
python validate_submission.py output/submission.csv

# (Optional) rebuild the embedding cache offline:
uv venv --python 3.12 .venv-embed
uv pip install --python .venv-embed torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-embed sentence-transformers numpy
.venv-embed/bin/python precompute/build_embeddings.py

# Offline evaluation:
python eval/evaluate.py
python tests/test_metrics.py
```

With `cache/embeddings.npz` present, `rank.py` runs the hybrid ranker;
without it, rule-only. Either path produces a valid submission.

---

## 8. What we'd do next

- **Grow + de-polarize the golden set.** The current 53 labels saturate the
  metric; adding "mushy middle" candidates (grade 2–3) is the only way to make
  the metric sensitive enough to actually tune the semantic weight and the
  behavioral floor.
- **Read career `description` text more deeply.** The semantic layer already
  embeds it, but a targeted extractive pass (e.g. pulling out concrete
  metrics like "0.72 → 0.91 NDCG@10") would be a strong, hard-to-fake signal.
- **Calibrate the behavioral floor** against real recruiter outcome data once
  any exists; right now 0.65 is a reasoned guess.
