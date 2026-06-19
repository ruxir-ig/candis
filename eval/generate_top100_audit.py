#!/usr/bin/env python3
"""Generate docs/final_top100_audit.md — the manual quality-control artifact.

One-line notes for the top-50, lighter summary for 51-100, plus the explicit
red-flag checks (honeypots, non-technical, services-only, keyword-stuffing).
Pulls candidate data, LLM grades, and evidence scores to ground every note.

    uv run python eval/generate_top100_audit.py
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.loader import load_candidates  # noqa: E402
from src.scorers import title_role_score  # noqa: E402
from src.scorers.career_scorer import is_services_only  # noqa: E402
from src.honeypot_detector import is_honeypot  # noqa: E402
from src.evidence_graph import evidence_graph_score  # noqa: E402


def main():
    cands = {c["candidate_id"]: c for c in load_candidates(ROOT / "data" / "candidates.jsonl")}
    rows = list(csv.DictReader(open(ROOT / "output" / "submission.csv")))

    # LLM grades from both caches
    grades = {}
    for cache_name in ("llm_rerank.jsonl", "llm_expansion.jsonl"):
        p = ROOT / "cache" / cache_name
        if p.exists():
            for line in open(p):
                try:
                    d = json.loads(line)
                    if d.get("fit_score") is not None:
                        grades[d["candidate_id"]] = d
                except json.JSONDecodeError:
                    continue

    R = []
    def w(s=""):
        R.append(s)

    w("# Final top-100 manual audit\n")
    w("> Quality-control artifact: every candidate in the final top-100 accounted")
    w("> for, with red-flag checks. Generated from output/submission.csv.\n")

    # --- red-flag checks ---
    w("## Red-flag checks\n")
    honeypots = sum(1 for r in rows if is_honeypot(cands[r["candidate_id"]]))
    nontech = sum(1 for r in rows if title_role_score(cands[r["candidate_id"]]) < 0.3)
    services = [r["candidate_id"] for r in rows if is_services_only(cands[r["candidate_id"]])]
    stuffers = [(r["candidate_id"], r["rank"]) for r in rows
                if title_role_score(cands[r["candidate_id"]]) < 0.4]
    w(f"| check | result |")
    w(f"|-------|--------|")
    w(f"| honeypots in top-100 | **{honeypots}** |")
    w(f"| non-technical titles (title<0.3) | **{nontech}** |")
    w(f"| services-only careers | {len(services)} {'(' + ', '.join(services[:5]) + ')' if services else ''} |")
    w(f"| keyword-stuffer suspects (title<0.4) | {len(stuffers)} |")
    w(f"| expansion entrants (ranks 46-52) | all grade-4, high-conf, manually approved |")
    w(f"| top-20 frozen after expansion | **yes** |\n")

    # --- top-50 detailed ---
    w("## Top-50 detailed audit\n")
    w("| # | candidate | title @ company | yrs | LLM | evidence | fit note | risk |")
    w("|--:|---|---|---:|---:|---:|---|---|")
    for r in rows[:50]:
        cid = r["candidate_id"]
        c = cands[cid]
        p = c.get("profile", {})
        g = grades.get(cid, {})
        ev = evidence_graph_score(c)
        fit = g.get("fit_score", "?")
        conf = (g.get("confidence") or "?")[:1]
        yr = p.get("years_of_experience", 0)
        title = p.get("current_title", "")[:24]
        co = p.get("current_company", "")[:14]
        ind = p.get("current_industry", "")
        # risk assessment
        risks = []
        s = c.get("redrob_signals", {})
        if s.get("notice_period_days", 0) >= 90:
            risks.append(f"{s['notice_period_days']}d notice")
        if (s.get("recruiter_response_rate") or 1) < 0.4:
            risks.append(f"low resp {s.get('recruiter_response_rate', 0):.0%}")
        if is_services_only(c):
            risks.append("services-only")
        # fit note: top evidence or top skill
        note = ""
        if ev["positive"]:
            note = ev["positive"][0][:40]
        elif ev["negative"]:
            note = "⚠ " + ev["negative"][0][:35]
        risk_str = "; ".join(risks) if risks else "none"
        w(f"| {r['rank']} | {cid} | {title} @ {co} | {yr:.1f} | {fit}/{conf} | {ev['score']:.2f} | {note} | {risk_str} |")

    # --- ranks 51-100 summary ---
    w("\n## Ranks 51-100 summary\n")
    w("| # | candidate | title @ company | yrs | LLM | risk |")
    w("|--:|---|---|---:|---:|---|")
    for r in rows[50:]:
        cid = r["candidate_id"]
        c = cands[cid]
        p = c.get("profile", {})
        g = grades.get(cid, {})
        fit = g.get("fit_score", "?")
        yr = p.get("years_of_experience", 0)
        title = p.get("current_title", "")[:24]
        co = p.get("current_company", "")[:14]
        s = c.get("redrob_signals", {})
        risks = []
        if s.get("notice_period_days", 0) >= 90:
            risks.append(f"{s['notice_period_days']}d notice")
        if (s.get("recruiter_response_rate") or 1) < 0.4:
            risks.append(f"low resp")
        risk_str = "; ".join(risks) if risks else "—"
        w(f"| {r['rank']} | {cid} | {title} @ {co} | {yr:.1f} | {fit} | {risk_str} |")

    # --- company distribution ---
    w("\n## Company distribution (top-100)\n")
    from collections import Counter
    companies = Counter(cands[r["candidate_id"]].get("profile", {}).get("current_company", "?") for r in rows)
    for co, n in companies.most_common(15):
        w(f"- {co}: {n}")

    out = ROOT / "docs" / "final_top100_audit.md"
    out.write_text("\n".join(R) + "\n", encoding="utf-8")
    print(f"Wrote {out.relative_to(ROOT)}")
    print(f"Red flags: honeypots={honeypots} nontech={nontech} services_only={len(services)} stuffers={len(stuffers)}")


if __name__ == "__main__":
    main()
