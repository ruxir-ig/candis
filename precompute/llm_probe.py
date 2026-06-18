"""Calibration probe: can the configured LLM judge candidates correctly?

Picks ~8 profiles where the right answer is unambiguous (clear stars, clear
traps, and one mushy case), has the LLM grade them, and compares to the
hand-labels in eval/golden_labels.csv. This is the empirical "is the model good
enough" test before we commit to labeling 500+ at scale.

Run:
    python precompute/llm_probe.py
Set the backend via env, e.g.:
    LLM_BASE_URL=https://api.groq.com/openai/v1 LLM_API_KEY=gsk_... \
    LLM_MODEL=llama-3.3-70b-versatile python precompute/llm_probe.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.loader import load_candidates  # noqa: E402
from src.llm_client import chat, config_summary  # noqa: E402
from src.llm_judge import judge_messages  # noqa: E402


def load_golden_labels() -> dict:
    labels = {}
    with open(ROOT / "eval" / "golden_labels.csv", encoding="utf-8") as f:
        for row in __import__("csv").DictReader(f):
            if row.get("grade", "").strip():
                labels[row["candidate_id"]] = int(row["grade"])
    return labels


def parse_grade(text: str) -> dict | None:
    """Robustly extract the JSON judgment, tolerating <think> + prose wrappers.

    Reasoning models (DeepSeek-R1, GLM, Qwen3) emit long <think> blocks before
    the answer. We strip those, then try a direct JSON parse, then fall back to
    the first {...} blob. If the answer itself got truncated (think block ate the
    whole budget), we return None so the caller can retry with more tokens.
    """
    if not text:
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", cleaned, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def main() -> int:
    labels = load_golden_labels()
    # Pick a stratified, unambiguous sample: the extreme grades (0s and 4s).
    by_grade = {0: [], 1: [], 2: [], 3: [], 4: []}
    for cid, g in labels.items():
        by_grade[g].append(cid)
    picks = (by_grade[4][:3] + by_grade[3][:1] + by_grade[2][:1]
             + by_grade[1][:1] + by_grade[0][:3])
    print(f"LLM backend: {config_summary()}")
    print(f"Probe size: {len(picks)} candidates (3 grade-4, 1 grade-3, 1 grade-2, 1 grade-1, 3 grade-0)\n")

    cands = {c["candidate_id"]: c for c in load_candidates(ROOT / "data" / "candidates.jsonl")
             if c["candidate_id"] in picks}

    # Reasoning models need a bigger budget (think block + answer); allow override.
    max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "1500"))
    # JSON-mode forcing can conflict with some reasoning models; make it optional.
    force_json = os.environ.get("LLM_FORCE_JSON", "1") not in ("0", "false", "no")

    exact = within1 = total = 0
    for cid in picks:
        c = cands.get(cid)
        if not c:
            continue
        total += 1
        truth = labels[cid]
        title = c.get("profile", {}).get("current_title", "")
        company = c.get("profile", {}).get("current_company", "")
        try:
            raw = chat(judge_messages(c), temperature=0.0, max_tokens=max_tokens,
                       response_format_json=force_json)
        except Exception as e:
            print(f"[{cid}] CALL FAILED: {e}")
            continue
        out = parse_grade(raw) or {}
        pred = out.get("grade")
        ok = "✓" if pred == truth else ("~" if pred is not None and abs(pred - truth) <= 1 else "✗")
        if pred == truth:
            exact += 1
        if pred is not None and abs(pred - truth) <= 1:
            within1 += 1
        print(f"{ok} {cid}  truth={truth} pred={pred}  conf={out.get('confidence','?')}")
        print(f"     {title} @ {company}")
        if out.get("reasoning"):
            print(f"     reasoning: {out['reasoning'].strip()[:240]}")
        elif raw:
            print(f"     [parse failed] raw tail: ...{raw[-220:].strip()}")
        print()

    if total:
        print(f"Exact agreement: {exact}/{total}   within-1: {within1}/{total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
