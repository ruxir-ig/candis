# 🎯 India Runs — Redrob Hackathon: Project Plan

> **Deadline:** July 2, 2026 23:59 IST  
> **Time remaining:** ~16 days  
> **Competition:** Intelligent Candidate Discovery & Ranking Challenge  
> **Dataset:** 100K candidates (JSONL, ~465MB) → rank top 100 for a Senior AI/ML Engineer JD

---

## 📋 What the Problem Actually Is

This isn't a keyword-matching exercise. The JD itself **explicitly warns against it**:

> *"The right answer is not 'find candidates whose skills section contains the most AI keywords.' That's a trap we've explicitly built into the dataset."*

What the JD actually wants:
1. **Senior AI/ML Engineer** (5-9 yrs) who has shipped **ranking/search/retrieval systems** in production
2. Must have **product company** experience (not just consulting/services)
3. Needs **embeddings, vector DBs, evaluation frameworks** experience
4. Location preference: **India (Noida/Pune/Hyderabad/Mumbai/Delhi NCR)**
5. Behavioral signals matter: **active, responsive, available candidates** rank higher
6. **~80 honeypot** candidates with impossible profiles — must be detected and eliminated

---

## 🏗️ Architecture: Multi-Stage Ranking Pipeline

```
┌──────────────────────────────────────────────────────────┐
│                    100K Candidates                       │
└─────────────────────────┬────────────────────────────────┘
                          │
                    ┌─────▼─────┐
                    │  STAGE 1  │  Honeypot Detection
                    │  Filter   │  (~80 removed)
                    └─────┬─────┘
                          │
                    ┌─────▼─────┐
                    │  STAGE 2  │  Hard Filters (experience range,
                    │  Coarse   │  country, domain mismatch)
                    │  Filter   │  → ~5K-15K survive
                    └─────┬─────┘
                          │
                    ┌─────▼─────┐
                    │  STAGE 3  │  Multi-Dimensional Scoring
                    │  Scoring  │  (career, skills, education,
                    │  Engine   │  title, behavioral signals)
                    └─────┬─────┘
                          │
                    ┌─────▼─────┐
                    │  STAGE 4  │  Semantic Matching
                    │  Semantic │  (sentence-transformers,
                    │  Reranker │  profile↔JD similarity)
                    └─────┬─────┘
                          │
                    ┌─────▼─────┐
                    │  STAGE 5  │  Final Ranking + Reasoning
                    │  Combine  │  (weighted ensemble → top 100)
                    └─────┬─────┘
                          │
                    ┌─────▼─────┐
                    │  OUTPUT   │  submission.csv
                    └───────────┘
```

---

## 🔬 Detailed Stage Breakdown

### Stage 1: Honeypot Detection
**What you'll learn:** Anomaly detection, data validation, logical reasoning

Honeypots have "subtly impossible profiles":
- 8 years experience at a company founded 3 years ago
- "Expert" proficiency in 10 skills with 0 `duration_months`
- Career history durations that don't add up
- Skill endorsements wildly inconsistent with experience level

**Implementation:**
```python
def detect_honeypot(candidate):
    flags = 0
    # Check: expert skills with 0 duration
    expert_zero = sum(1 for s in skills if s['proficiency'] == 'expert' and s.get('duration_months', 0) == 0)
    if expert_zero >= 3: flags += 1
    
    # Check: career history total >> stated years_of_experience
    total_career = sum(r['duration_months'] for r in career_history)
    if abs(total_career / 12 - years_of_experience) > 3: flags += 1
    
    # Check: impossibly high skill count for experience level
    # ... more checks
    return flags >= 2
```

### Stage 2: Coarse Filters (Rule-Based)
**What you'll learn:** Feature engineering, domain modeling

Hard filters extracted **from reading the JD carefully**:
- **Experience:** 3-15 years (generous band around 5-9)
- **Domain:** Must have some AI/ML/Data/Engineering in career — filter out pure Marketing, HR, Accounting, etc.
- **Country/Location:** Prioritize India; penalize (don't eliminate) non-India
- **Career pattern:** Must have at least 1 product-company role (not just consulting firms)
- **Title relevance:** Current or recent title should be engineering/technical

### Stage 3: Multi-Dimensional Scoring Engine
**What you'll learn:** Scoring system design, feature weighting, normalization

Seven scoring dimensions (each 0-1):

| Dimension | Weight | What it captures |
|-----------|--------|------------------|
| **Title & Role Fit** | 0.20 | How close are their titles to "AI/ML Engineer", "Data Scientist", etc. |
| **Skills Match** | 0.20 | Semantic skill matching against JD requirements (embeddings, vector DBs, Python, evaluation) |
| **Career Quality** | 0.15 | Product company experience, career progression, duration patterns |
| **Experience Fit** | 0.10 | Optimal range 5-9 yrs, decay outside |
| **Education** | 0.05 | Tier weighting (not dominant — JD doesn't emphasize it) |
| **Location Fit** | 0.05 | India, specific cities, relocation willingness |
| **Behavioral Signals** | 0.25 | The **most important modifier** — availability, responsiveness, platform activity |

#### Skills Match — Deep Dive
The JD lists explicit skill tiers:

**Must-have skills (highest weight):**
- Embeddings-based retrieval (sentence-transformers, BGE, E5)
- Vector databases (Pinecone, Weaviate, Qdrant, FAISS, Elasticsearch)
- Strong Python
- Ranking evaluation (NDCG, MRR, MAP, A/B testing)

**Nice-to-have skills:**
- LLM fine-tuning (LoRA, QLoRA, PEFT)
- Learning-to-rank models
- HR-tech / recruiting tech
- Distributed systems

**Red flag skills (anti-signal):**
- Only LangChain + OpenAI API (recent, shallow)
- Pure computer vision / robotics

#### Behavioral Signal Scoring
```python
def behavioral_score(signals):
    score = 0.0
    
    # Availability (40% of behavioral)
    if signals['open_to_work_flag']: score += 0.10
    days_since_active = (today - signals['last_active_date']).days
    score += max(0, 0.15 * (1 - days_since_active / 180))
    if signals['recruiter_response_rate'] > 0.5: score += 0.15
    
    # Engagement quality (30% of behavioral)
    score += 0.10 * min(1, signals['profile_completeness_score'] / 80)
    score += 0.10 * min(1, signals['interview_completion_rate'])
    score += 0.10 * signals.get('offer_acceptance_rate', 0) if signals.get('offer_acceptance_rate', -1) > 0 else 0
    
    # Notice period (15% of behavioral)
    if signals['notice_period_days'] <= 30: score += 0.15
    elif signals['notice_period_days'] <= 60: score += 0.08
    
    # Platform trust (15% of behavioral)
    if signals['verified_email']: score += 0.05
    if signals['verified_phone']: score += 0.05
    if signals['linkedin_connected']: score += 0.05
    
    return score
```

### Stage 4: Semantic Matching (Embeddings)
**What you'll learn:** Sentence transformers, vector similarity, embedding-based search

Pre-compute embeddings for:
- The JD text (requirements section)
- Each candidate's combined profile text (summary + career descriptions)

Use `sentence-transformers` (e.g., `all-MiniLM-L6-v2` for speed, or `all-mpnet-base-v2` for quality):
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# Pre-compute (can be done offline, outside 5-min window)
jd_embedding = model.encode(jd_text)
candidate_embeddings = model.encode([c.profile_text for c in candidates])

# At ranking time: cosine similarity
similarities = cosine_similarity([jd_embedding], candidate_embeddings)[0]
```

### Stage 5: Final Ensemble + Reasoning
**What you'll learn:** Score calibration, ensemble methods, explainability

```
final_score = (
    0.35 * rule_based_score +      # Stages 2-3
    0.35 * semantic_score +         # Stage 4
    0.30 * behavioral_modifier      # Stage 3 behavioral
)
```

Generate 1-2 sentence reasoning for each candidate:
```
"Senior ML Engineer at [ProductCo] with 7.2 yrs experience; 
production embeddings + vector DB expertise; 
active on platform (response rate 0.82, 15-day notice)."
```

---

## 🗂️ Repo Structure (Clean, Competition-Ready)

```
india-runs/
├── README.md                          # Setup, reproduce, architecture overview
├── requirements.txt                   # Pinned dependencies
├── submission_metadata.yaml           # Filled-in metadata
├── rank.py                            # Single entry point: python rank.py → submission.csv
│
├── src/                               # Core pipeline modules
│   ├── __init__.py
│   ├── config.py                      # All constants, weights, thresholds
│   ├── loader.py                      # Load candidates.jsonl efficiently
│   ├── jd_parser.py                   # Parse and structure the JD
│   ├── honeypot_detector.py           # Stage 1: anomaly detection
│   ├── coarse_filter.py               # Stage 2: hard filters
│   ├── scorers/                       # Stage 3: scoring modules
│   │   ├── __init__.py
│   │   ├── title_scorer.py
│   │   ├── skills_scorer.py
│   │   ├── career_scorer.py
│   │   ├── experience_scorer.py
│   │   ├── education_scorer.py
│   │   ├── location_scorer.py
│   │   └── behavioral_scorer.py
│   ├── semantic_matcher.py            # Stage 4: embedding-based matching
│   ├── ranker.py                      # Stage 5: ensemble + final ranking
│   └── reasoning.py                   # Generate human-readable reasoning
│
├── precompute/                        # Offline pre-computation scripts
│   ├── build_embeddings.py            # Generate & cache embeddings
│   └── analyze_dataset.py             # EDA / dataset analysis
│
├── data/                              # Dataset files (gitignored, not the raw zip)
│   ├── candidates.jsonl               # ← gitignored (too large)
│   └── job_description.txt            # Extracted JD text
│
├── cache/                             # Pre-computed artifacts (committed)
│   └── embeddings.npz                 # Cached embeddings
│
├── notebooks/                         # Exploration & analysis
│   ├── 01_eda.ipynb                   # Data exploration
│   ├── 02_honeypot_analysis.ipynb     # Honeypot investigation
│   └── 03_scoring_tuning.ipynb        # Weight tuning experiments
│
├── output/                            # Generated outputs
│   └── submission.csv                 # Final submission
│
├── tests/                             # Validation & testing
│   ├── test_honeypot.py
│   ├── test_scorers.py
│   └── test_pipeline.py
│
├── docs/                              # Documentation
│   └── approach.md                    # Detailed methodology write-up
│
├── sandbox/                           # Streamlit app for demo
│   └── app.py                         # Streamlit sandbox UI
│
├── validate_submission.py             # Official validator (from bundle)
├── .gitignore
└── LICENSE
```

---

## 📅 16-Day Timeline

### Phase 1: Foundation (Days 1-3, June 16-18)
- [x] Understand the problem deeply ← **you are here**
- [ ] Clean & organize repo structure
- [ ] Write data loader + EDA notebook
- [ ] Extract & structure the JD
- [ ] Implement honeypot detection
- [ ] Implement coarse filters

### Phase 2: Scoring Engine (Days 4-7, June 19-22)
- [ ] Build all 7 scorer modules
- [ ] Title scorer (with fuzzy matching)
- [ ] Skills scorer (semantic skill matching)
- [ ] Career scorer (product vs services, progression)
- [ ] Behavioral scorer
- [ ] Experience, education, location scorers
- [ ] Validate against sample candidates manually

### Phase 3: Semantic Layer (Days 8-10, June 23-25)
- [ ] Set up sentence-transformers
- [ ] Pre-compute embeddings (offline)
- [ ] Build semantic matcher
- [ ] Integrate with rule-based scores

### Phase 4: Ensemble & Tuning (Days 11-13, June 26-28)
- [ ] Build final ensemble ranker
- [ ] Manual inspection of top-100 rankings
- [ ] Tune weights based on sanity checks
- [ ] Generate reasoning strings
- [ ] Validate submission format

### Phase 5: Polish & Submit (Days 14-16, June 29-July 1)
- [ ] Build Streamlit sandbox
- [ ] Write README.md with full instructions
- [ ] Write approach document/deck (PDF)
- [ ] Clean git history (show real iteration)
- [ ] Final validation + submit

---

## 🧠 What You'll Learn (Maximized)

| Concept | Where it shows up |
|---------|-------------------|
| **Data Engineering** | Loading 465MB JSONL efficiently, streaming processing |
| **Anomaly Detection** | Honeypot identification via statistical inconsistencies |
| **Feature Engineering** | Designing 7 scoring dimensions from raw candidate data |
| **NLP / Embeddings** | Sentence-transformers, cosine similarity, semantic search |
| **Information Retrieval** | NDCG, MAP, precision@K evaluation metrics |
| **Scoring System Design** | Multi-factor ranking, weight tuning, normalization |
| **LLM Understanding** | When to use embeddings vs keywords vs LLMs |
| **Production ML** | Compute constraints (5min, 16GB, CPU-only, no network) |
| **Explainable AI** | Generating human-readable reasoning for each ranking |
| **MLOps / Reproducibility** | Single-command reproduction, dependency pinning |
| **Software Engineering** | Clean repo, modular code, testing, documentation |

---

## ⚠️ Critical Gotchas

> [!CAUTION]
> **Honeypots:** ~80 candidates with impossible profiles. More than 10 in your top 100 = disqualified.

> [!CAUTION]
> **Keyword stuffing trap:** Marketing Managers with 9 "AI core skills" are NOT good candidates. The JD explicitly warns about this.

> [!WARNING]
> **Compute constraint:** Ranking step must complete in 5 minutes on CPU with 16GB RAM and no network. Pre-computation (embeddings) can happen offline.

> [!WARNING]
> **Consulting firm trap:** The JD says candidates who've ONLY worked at TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini are "not a fit."

> [!IMPORTANT]
> **The JD gap:** "The right answer involves reasoning about the gap between what the JD says and what the JD means." A candidate who built a recommendation system at a product company is a fit even without the words "RAG" or "Pinecone."

> [!NOTE]
> **Scoring formula:** `0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10` — getting the **top 10 right is worth 50%** of the score.

---

## 🚀 Let's Build This

Ready to start with Phase 1? I'll reorganize the repo, set up the project structure, and build the data loader + honeypot detector first.
