"""Turn a candidate record into the text we embed.

The rule-based scorers read structured fields (title enum, skill list). The
embedding instead reads the *prose* — headline, summary, and what the person
actually did in each role — because that's where a genuine fit reveals itself
("rebuilt the candidate-JD matching pipeline ... 0.72 -> 0.91 NDCG@10") and
where a keyword-stuffer gives itself away (an HR manager's summary is about HR,
no matter what skills they pasted on).

Kept compact: all-MiniLM-L6-v2 truncates around 256 word-pieces, so we lead with
the most informative text (summary, recent role descriptions).
"""

MAX_ROLES = 4
MAX_DESC_CHARS = 400
MAX_TOTAL_CHARS = 1400


def candidate_text(candidate: dict) -> str:
    p = candidate.get("profile", {})
    parts = []

    headline = p.get("headline")
    if headline:
        parts.append(headline)

    summary = p.get("summary")
    if summary:
        parts.append(summary)

    for role in candidate.get("career_history", [])[:MAX_ROLES]:
        title = role.get("title", "")
        company = role.get("company", "")
        desc = (role.get("description", "") or "")[:MAX_DESC_CHARS]
        line = f"{title} at {company}. {desc}".strip()
        if line:
            parts.append(line)

    skills = [s.get("name", "") for s in candidate.get("skills", []) if s.get("name")]
    if skills:
        parts.append("Skills: " + ", ".join(skills[:15]))

    return " \n".join(parts)[:MAX_TOTAL_CHARS]
