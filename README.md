# 🎯 Intelligent Candidate Discovery & Ranking Challenge

An AI-driven capability-matching pipeline built for the Redrob Intelligent Candidate Discovery & Ranking Challenge. The system moves beyond keyword-matching to accurately discover, rank, and shortlist the top 100 candidates for a Senior AI/ML Engineer role out of a pool of 100,000 candidates.

---

## 🏗️ Repository Architecture

The project has been organized into a clean, modular repository structure:

```
india-runs/
├── README.md                          # This file: Setup, reproduction, architecture overview
├── requirements.txt                   # Pinned python dependencies
├── submission_metadata_template.yaml   # Template for portal metadata
├── submission_metadata.yaml           # Active submission metadata (copy of template)
├── validate_submission.py             # Official submission format validator
│
├── src/                               # Core pipeline modules
│   ├── __init__.py
│   ├── loader.py                      # Stage 0: Efficient JSONL/GZIP data loading
│   └── honeypot_detector.py           # Stage 1: Logical anomaly & honeypot filtering
│
├── precompute/                        # Offline analysis & caching scripts
│   └── analyze_dataset.py             # Dataset EDA & anomaly profiling
│
├── data/                              # Datasets (gitignored except for format files)
│   ├── candidate_schema.json          # Schema definition
│   └── sample_candidates.json         # First 50 candidates preview
│
├── docs/                              # Original challenge documentation
│   ├── README.docx                    # Getting started doc
│   ├── job_description.docx           # Senior AI/ML Engineer JD requirements
│   ├── redrob_signals_doc.docx        # Behavioral signals explanation
│   └── submission_spec.docx           # Submission rules and rules guidelines
│
├── cache/                             # Cache directory for pre-computed artifacts (gitignored)
├── output/                            # Target directory for generated submissions (gitignored)
├── notebooks/                         # Jupyter notebooks for experimentation
└── tests/                             # Unit tests & verification
    └── test_honeypot.py               # Verify loader & honeypot filtering
```

---

## ⚡ Setup & Installation

### 1. Prerequisites
- **Python 3.11+**
- **pip**

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Place Datasets
Unpack the zipped dataset and place the files inside the `data/` directory:
- `data/candidates.jsonl` (raw or gzipped as `candidates.jsonl.gz`)

---

## 🧪 Running Verification

To run the verification test for the data loader and honeypot detection:

```bash
python3 tests/test_honeypot.py
```

---

## 🔬 Next Steps

We are executing this project in **5 key phases**:

1. **Phase 1: Foundation (Current)** - Setup project structure, data loader, and logical honeypot detection.
2. **Phase 2: Multi-Dimensional Scoring Engine** - Implement individual scorers (skills, experience, education, title matching, and behavioral signals).
3. **Phase 3: Semantic Layer** - Integrate sentence embeddings (`sentence-transformers`) for deep semantic job-to-profile alignment.
4. **Phase 4: Ensemble Ranking & Calibration** - Build the final composite ranking function, calibrate weights, and write explanation generators.
5. **Phase 5: Submission & Polish** - Construct the HuggingFace Spaces/Streamlit sandbox app and compile the methodology slide deck.
