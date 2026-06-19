# Loop 5 Plan — Fairness / Responsible AI Audit

**Date started:** 19 June 2026  
**Current loop number:** 5  
**Submission deadline:** 2 July 2026, 23:59 IST  
**Status:** Complete

---

## Goal

Audit whether the final top-100 ranking is overly dependent on proxies (education tier, company brand, location, experience band, availability). Verify the system focuses on job-relevant evidence, not privilege signals.

## Checks

1. **Education concentration** — Is the system just ranking IIT/elite-school candidates? (Education weight = 0.05)
2. **Company-brand concentration** — Is the system overfitting to FAANG/big-tech?
3. **Location concentration** — Is location effectively a hard filter? (It's soft.)
4. **Experience distribution** — Does the 5-9yr sweet spot dominate excessively?
5. **Availability influence** — Does availability rescue weak-fit candidates? (It's a multiplier, not additive.)
6. **Component sensitivity** — Would removing education/location/availability significantly change top-100?

## Adoption gates

Audit-only. Only change ranking if a serious proxy-risk issue is found.

## Results

- Education: tier-1 enriched 12.6x but weight is 5%; 75% of top-100 stable without it
- Big-tech: 32% of top-100 (pool is tiny — only 39 big-tech candidates in 83K)
- Location: NOT a barrier — 69% of top-100 are non-India despite India preference
- Experience: 87% in 5-9y band (JD sweet spot), 13% under 5y, 0% above 9y
- Availability: +0.049 effect (under 0.05 threshold) — cannot rescue weak fit
- Component sensitivity: education 25 seats, location 16, experience 19 — no single component dominates

## Decision

AUDIT-ONLY. No fairness red flags. System focuses on job-relevant evidence.
