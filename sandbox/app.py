"""Redrob Candidate Ranking — interactive sandbox.

A read-only explorer over the submission: the top-100 ranked candidates with
their full score breakdown and profile. Run it with:

    pip install streamlit pandas
    streamlit run sandbox/app.py

It loads output/submission.csv (the ranked top-100) and joins each row to the
candidate's full record in data/candidates.jsonl so you can inspect *why* the
ranker placed someone where it did — component scores, skills, career history,
and behavioral signals. No re-scoring happens here; the sandbox only displays
what rank.py already produced.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
SUBMISSION = ROOT / "output" / "submission.csv"
CANDIDATES = ROOT / "data" / "candidates.jsonl"


@st.cache_data(show_spinner=False)
def load_submission() -> pd.DataFrame:
    return pd.read_csv(SUBMISSION).sort_values("rank").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_candidates_index() -> dict:
    idx = {}
    with open(CANDIDATES, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            idx[c["candidate_id"]] = c
    return idx


def best_title_company(c: dict) -> tuple[str, str]:
    prof = c.get("profile", {})
    title = prof.get("current_title") or ""
    company = prof.get("current_company") or ""
    if not title and (c.get("career_history") or []):
        title = c["career_history"][0].get("title", "")
        company = c["career_history"][0].get("company", "")
    return title, company


def main() -> None:
    st.set_page_config(page_title="India Runs — Candidate Ranker", layout="wide")
    st.title("Intelligent Candidate Discovery & Ranking")
    st.caption("Senior AI/ML Engineer — top 100 of 100,000 candidates")

    sub = load_submission()
    cands = load_candidates_index()

    # ---- Leaderboard table -------------------------------------------------
    rows = []
    for _, r in sub.iterrows():
        c = cands.get(r["candidate_id"], {})
        title, company = best_title_company(c)
        rows.append({
            "rank": r["rank"],
            "candidate_id": r["candidate_id"],
            "score": r["score"],
            "title": title,
            "company": company,
        })
    table = pd.DataFrame(rows)

    st.subheader("Ranked top 100")
    sel = st.radio(
        "View",
        ["Leaderboard", "Inspect a candidate", "Honeypot / trap audit"],
        horizontal=True,
    )

    if sel == "Leaderboard":
        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "score": st.column_config.ProgressColumn(
                    "score", help="final = fit × availability",
                    format="%.4f", min_value=0.0, max_value=1.0,
                ),
            },
        )
        st.caption(f"{len(table)} candidates · top score {table['score'].max():.4f} · "
                   f"median {table['score'].median():.4f}")
        return

    if sel == "Inspect a candidate":
        cid = st.selectbox("Candidate", table["candidate_id"], format_func=lambda i: i)
        show_candidate(cid, cands, sub)
        return

    # Honeypot / trap audit
    show_trap_audit(cands, sub)


def show_candidate(cid: str, cands: dict, sub: pd.DataFrame) -> None:
    c = cands.get(cid, {})
    row = sub[sub["candidate_id"] == cid].iloc[0]
    title, company = best_title_company(c)
    prof = c.get("profile", {})
    sig = c.get("redrob_signals", {})

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rank", f"#{int(row['rank'])}")
    m2.metric("Final score", f"{row['score']:.4f}")
    m3.metric("Years exp", prof.get("years_of_experience", "—"))
    m4.metric("Location", prof.get("location", "—"))

    st.markdown(f"### {title} @ {company}")
    st.caption(f"`{cid}`")
    st.markdown(row["reasoning"])

    # Component breakdown — recompute live so the chart always reflects the
    # current config weights without needing the score stored per component.
    comp = score_components(c)
    st.subheader("Score breakdown")
    cleft, cright = st.columns([3, 2])
    with cleft:
        comp_df = pd.DataFrame(
            [{"component": k, "value [0-1]": v} for k, v in comp.items()]
        )
        st.bar_chart(comp_df, x="component", y="value [0-1]", use_container_width=True)
    with cright:
        st.dataframe(comp_df.set_index("component"), use_container_width=True)

    # Skills
    st.subheader("Skills")
    sk = c.get("skills", []) or []
    if sk:
        sk_df = pd.DataFrame([{
            "skill": s.get("name", ""),
            "proficiency": s.get("proficiency", ""),
            "months": s.get("duration_months", 0),
        } for s in sk])
        st.dataframe(sk_df, use_container_width=True, hide_index=True)
    else:
        st.write("_none listed_")

    # Behavioral signals
    st.subheader("Behavioral signals")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Response rate", f"{(sig.get('recruiter_response_rate') or 0)*100:.0f}%")
    b2.metric("Notice", f"{sig.get('notice_period_days', '—')} d")
    b3.metric("Last active", str(sig.get("last_active_date", "—")))
    b4.metric("Profile complete", f"{sig.get('profile_completeness_score', '—')}/100")

    # Career history
    st.subheader("Career history")
    hist = c.get("career_history", []) or []
    for h in hist:
        cur = " (current)" if h.get("is_current") else ""
        dur = h.get("duration_months", 0)
        st.markdown(f"- **{h.get('title','')}** @ {h.get('company','')}{cur} — {dur} mo")
        if h.get("description"):
            st.caption(h["description"][:300])


def score_components(c: dict) -> dict:
    from datetime import date
    from src.ranker import reference_date
    from src.scorers import (
        title_role_score, skills_score, career_score, experience_score,
        education_score, location_score, behavioral_availability,
    )
    from src.semantic_matcher import semantic_scores

    sem = semantic_scores() or {}
    ref = reference_date([c])
    return {
        "title": title_role_score(c),
        "skills": skills_score(c),
        "career": career_score(c),
        "experience": experience_score(c),
        "education": education_score(c),
        "location": location_score(c),
        "semantic": sem.get(c["candidate_id"], 0.0),
        "availability": behavioral_availability(c, ref),
    }


def show_trap_audit(cands: dict, sub: pd.DataFrame) -> None:
    st.subheader("Trap audit")
    from src.honeypot_detector import is_honeypot
    from src.coarse_filter import passes_coarse_filter

    top_ids = set(sub["candidate_id"])
    st.write("Sanity checks on the submitted top-100:")
    in_top = [cands[i] for i in top_ids if i in cands]
    n_honey = sum(1 for c in in_top if is_honeypot(c))
    n_trap = sum(1 for c in in_top if not passes_coarse_filter(c))
    a1, a2 = st.columns(2)
    a1.metric("Honeypots in top-100", n_honey, help="Must be 0.")
    a2.metric("Non-technical careers in top-100", n_trap, help="Must be 0.")

    st.markdown("**Title distribution in the top-100**")
    titles = []
    for c in in_top:
        t = (c.get("profile", {}).get("current_title") or "").lower()
        titles.append(t or "(unknown)")
    st.dataframe(pd.Series(titles).value_counts().head(15).rename_axis("title").reset_index(name="count"),
                 use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
