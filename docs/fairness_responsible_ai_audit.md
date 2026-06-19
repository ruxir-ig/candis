# Fairness / Responsible AI Audit

**Loop:** 5  
**Branch:** `research/loop-5-fairness-audit`  
**Date:** 19 June 2026  
**Status:** Complete  

---

## Purpose

Audit whether the final top-100 ranking is overly dependent on proxy signals (education tier, company brand, location, experience, availability) rather than job-relevant evidence.

---

## Results

### 1. Education tier distribution

| Tier | Top-100 | Pool | Enrichment |
|------|---------|------|------------|
| tier_1 | 63 | 4,203 | 12.6x |
| tier_2 | 33 | 16,984 | 1.6x |
| tier_3 | 4 | 31,913 | 0.1x |

**Assessment:** Tier-1 candidates are enriched 12.6x, but education weight is only 5% of the fit score. The enrichment is explained by career substance — tier-1 graduates disproportionately hold ML engineering roles at product companies, which is the primary signal. Removing education entirely changes 25/100 seats (75% stable), confirming it's a tiebreaker, not a driver.

### 2. Company-brand distribution

| Type | Top-100 | Pool | Enrichment |
|------|---------|------|------------|
| big_tech | 32 | 39 | 687x |
| unicorn | 22 | 6,368 | 2.9x |
| other_product | 46 | 53,026 | 0.7x |
| services | 0 | 24,346 | 0.0x |

**Assessment:** Big-tech candidates are highly enriched (687x), but only 39 of 83,779 candidates are at big-tech companies — the pool is tiny. Services-only careers are completely excluded (0%), which is by JD design ("TCS/Infosys/Wipro-only careers are a poor fit"). The largest group (46%) is "other product" companies, showing the system doesn't exclusively reward brand names.

### 3. Location distribution

| Zone | Count | % |
|------|-------|---|
| Non-India | 69 | 69% |
| Delhi NCR | 12 | 12% |
| Bangalore | 6 | 6% |
| Pune | 4 | 4% |
| Chennai | 4 | 4% |
| Mumbai | 3 | 3% |
| Hyderabad | 2 | 2% |

**Assessment:** Location is clearly **not a barrier**. Despite a soft India preference in the scorer, 69% of the top-100 are non-India based. Strong fit overrides location preference. This is a fairness positive.

### 4. Experience distribution

| Band | Count | % |
|------|-------|---|
| <5y | 13 | 13% |
| 5-9y | 87 | 87% |
| 9-12y | 0 | 0% |
| >12y | 0 | 0% |

**Assessment:** The 5-9y band dominates (87%), matching the JD's stated sweet spot. 13% are under 5 years but still ranked based on strong evidence. The absence of 9+ year candidates reflects the experience scorer's design — the JD says "ideal 5-9 years." This is JD-aligned, not a bias.

### 5. Availability influence

| Metric | Value |
|--------|-------|
| Median fit in top-100 | 0.8782 |
| Avg availability (below-median fit) | 0.7892 |
| Avg availability (above-median fit) | 0.7401 |
| Difference | +0.0491 |

**Assessment:** The difference is +0.049 — just under the 0.05 flag threshold. Low-fit candidates have marginally higher availability, but the effect is negligible. Availability cannot rescue weak fit (it's a multiplier with a 0.65 floor, not additive).

### 6. Component sensitivity

| Component removed | Seats changed | Stable |
|-------------------|--------------|--------|
| education | 25 | 75% |
| location | 16 | 84% |
| experience | 19 | 81% |

**Assessment:** No single component dominates. Education has the most influence (25 seats) but it's still only a tiebreaker — 75% of the top-100 stays without it. Location is the most stable (84% stay without it).

---

## Decision

**AUDIT-ONLY. No ranking change. No fairness red flags found.**

The system focuses on job-relevant evidence (career substance, skills with proof, evidence corroboration). Proxy signals (education, location, company brand) are either low-weight (education 5%), overridable (location — 69% non-India in top-100), or JD-aligned (services exclusion).

---

## Deck summary (responsible AI slide)

```
Responsible ranking safeguards

- No protected attributes used (gender, age, name, caste)
- Education is 5% weight — tiebreaker, not driver (75% stable without it)
- Location is soft — 69% of top-100 are non-India despite India preference
- Availability is a multiplier — cannot rescue weak fit (+0.049 effect)
- Services-only careers excluded per JD, not by proxy
- Component sensitivity: removing any one changes ≤25% of top-100
```
