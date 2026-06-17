"""Build a small, diverse set of candidates for YOU to hand-label.

Why: there is no leaderboard and no ground truth. So we manufacture our own
small "qrel" — you read ~55 profiles like a recruiter and grade each 0-4. That
labeled set becomes the feedback loop for every weight change afterwards.

The sample is stratified to include the cases that actually stress the system:
  - the top of the current ranking (where 50% of the score is decided)
  - "trap suspects": many AI skills but a non-technical title (the keyword
    stuffers the JD warns about) — the most informative profiles to label
  - mid-pool and random profiles (should mostly be irrelevant)
  - a few honeypots (impossible profiles; grade them 0)

Outputs:
  eval/golden_set_to_label.md   <- read this; full readable profiles
  eval/golden_labels.csv        <- fill the `grade` column (0-4), leave notes
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.loader import load_candidates  # noqa: E402
from src.ranker import score_all  # noqa: E402
from src.honeypot_detector import is_honeypot  # noqa: E402
from src.scorers import matched_skill_names  # noqa: E402

SEED = 42
GRADE_GUIDE = (
    "GRADE SCALE (label by reading the profile, not the system rank):\n"
    "  4 = excellent, clear top pick   3 = good, would shortlist (>=3 is 'relevant')\n"
    "  2 = borderline / maybe          1 = weak, probably not\n"
    "  0 = no / wrong domain / honeypot\n"
)


def profile_md(c: dict, tag: str, sysrank) -> str:
    p = c["profile"]
    s = c.get("redrob_signals", {})
    lines = [
        f"### {c['candidate_id']}  ·  [{tag}]  ·  system_rank={sysrank}",
        f"- **{p.get('current_title')}** at {p.get('current_company')} "
        f"({p.get('current_company_size')}, {p.get('current_industry')})",
        f"- {p.get('years_of_experience')} yrs · {p.get('location')}, {p.get('country')}",
        f"- Summary: {(p.get('summary') or '')[:280]}",
    ]
    hist = c.get("career_history", [])
    if hist:
        roles = "; ".join(
            f"{r.get('title')}@{r.get('company')}({r.get('duration_months')}mo)"
            for r in hist[:4]
        )
        lines.append(f"- Career: {roles}")
    rel = matched_skill_names(c, k=8)
    allsk = [sk.get("name") for sk in c.get("skills", [])][:12]
    lines.append(f"- Relevant skills: {', '.join(rel) if rel else '(none matched)'}")
    lines.append(f"- All skills: {', '.join(x for x in allsk if x)}")
    lines.append(
        f"- Signals: response={s.get('recruiter_response_rate')}, "
        f"last_active={s.get('last_active_date')}, notice={s.get('notice_period_days')}d, "
        f"open_to_work={s.get('open_to_work_flag')}, "
        f"github={s.get('github_activity_score')}, "
        f"interview_completion={s.get('interview_completion_rate')}"
    )
    return "\n".join(lines)


def main():
    rng = random.Random(SEED)
    print("Loading + scoring candidates ...")
    candidates = list(load_candidates(ROOT / "data/candidates.jsonl"))
    scored, _ = score_all(candidates)  # honeypots already dropped here
    rank_of = {r["candidate_id"]: i + 1 for i, r in enumerate(scored)}
    comp = {r["candidate_id"]: r for r in scored}

    picks = []  # (candidate, tag, sysrank)
    used = set()

    def add(c, tag):
        cid = c["candidate_id"]
        if cid in used:
            return
        used.add(cid)
        picks.append((c, tag, rank_of.get(cid, "n/a")))

    # 1) Top of the ranking — the decisions that dominate the score.
    for r in scored[:18]:
        add(r["candidate"], "top")

    # 2) Trap suspects: strong skills score but weak title score.
    traps = sorted(
        scored,
        key=lambda r: r["components"]["skills"] - r["components"]["title"],
        reverse=True,
    )
    for r in traps[:10]:
        add(r["candidate"], "trap-suspect")

    # 3) Mid-pool borderline (rank ~150-1500).
    mid = [r for r in scored[150:1500]]
    for r in rng.sample(mid, min(10, len(mid))):
        add(r["candidate"], "mid")

    # 4) Random broad pool (mostly should be irrelevant).
    for r in rng.sample(scored[3000:], min(10, len(scored[3000:]))):
        add(r["candidate"], "random")

    # 5) A few honeypots — confirm they're correctly excluded (grade 0).
    honeypots = [c for c in candidates if is_honeypot(c)]
    for c in rng.sample(honeypots, min(5, len(honeypots))):
        add(c, "honeypot")

    # Write the readable sheet.
    md = ROOT / "eval/golden_set_to_label.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write(f"# Golden set to label ({len(picks)} candidates)\n\n")
        f.write("```\n" + GRADE_GUIDE + "```\n\n")
        f.write("Read each profile, then put a grade in `eval/golden_labels.csv`.\n\n")
        for c, tag, sr in picks:
            f.write(profile_md(c, tag, sr) + "\n\n---\n\n")

    # Write the labels template (grade column blank for you to fill).
    csv_path = ROOT / "eval/golden_labels.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("candidate_id,grade,tag,system_rank,notes\n")
        for c, tag, sr in picks:
            f.write(f"{c['candidate_id']},,{tag},{sr},\n")

    print(f"Wrote {len(picks)} candidates:")
    print(f"  - read:  {md}")
    print(f"  - label: {csv_path}  (fill the 'grade' column 0-4)")


if __name__ == "__main__":
    main()
