"""Offline: grade a stratified sample of candidates with the LLM judge.

Produces cache/llm_labels.jsonl — a de-polarized qrel covering the full
relevance spectrum (stars, the mushy middle, hidden gems, and trap suspects).
This is the label set that finally makes our offline metric sensitive enough
to tune against (the 53 hand-labels saturate it).

Resumable: appends one JSON line per graded candidate and skips ids already
present, so an interrupted run (rate limits, network) just continues.

Sample modes:
  --sample eval     stratified by ranker rank + trap suspects + random (~400)
  --sample topN     the ranker's top N (for LLM re-ranking)
  --ids <file>      one candidate_id per line
  --n               sample size (default 400 for eval, 200 for topN)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.loader import load_candidates  # noqa: E402
from src.ranker import score_all  # noqa: E402
from src.llm_judge import judge_messages  # noqa: E402
from src.llm_client import chat, config_summary, LLMError  # noqa: E402

OUT = ROOT / "cache" / "llm_labels.jsonl"
MAX_TOKENS_DEFAULT = 1500


def already_done(path: Path) -> set:
    """Only count successfully-graded ids as done — error records get retried."""
    done = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("grade") is not None:
                    done.add(rec["candidate_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def build_eval_sample(ranked, cands_by_id, n=400, seed=13):
    """Stratified by rank + trap suspects + random, for a de-polarized qrel."""
    rnd = random.Random(seed)
    ids_in_order = [r["candidate_id"] for r in ranked]
    n_total = len(ids_in_order)

    def bucket(lo, hi):  # inclusive rank range, 1-indexed
        return [cands_by_id[i] for i in ids_in_order[lo - 1:hi]]

    sample = []
    # Stars / strong (ranks 1-50)
    sample += rnd.sample(bucket(1, 50), min(50, len(bucket(1, 50))))
    # Mushy middle high (51-200)
    sample += rnd.sample(bucket(51, 200), min(80, len(bucket(51, 200))))
    # Mushy middle low + potential hidden gems (201-1000)
    sample += rnd.sample(bucket(201, 1000), min(80, len(bucket(201, 1000))))
    # Deeper tail (1001-5000)
    sample += rnd.sample(bucket(1001, 5000), min(60, len(bucket(1001, 5000))))
    # Trap suspects: high skills score but weak title (the keyword stuffers)
    suspects = [r["candidate"] for r in ranked
                if r["components"]["skills"] > 0.6 and r["components"]["title"] < 0.3]
    sample += rnd.sample(suspects, min(80, len(suspects)))
    # Random from the whole pool (catches anything stratification missed)
    sample += rnd.sample(list(cands_by_id.values()), min(50, n_total))

    # Dedup preserving order, cap at n
    seen = set()
    out = []
    for c in sample:
        if c["candidate_id"] in seen:
            continue
        seen.add(c["candidate_id"])
        out.append(c)
        if len(out) >= n:
            break
    return out


def grade_one(candidate, model, force_json):
    raw = chat(judge_messages(candidate), temperature=0.0,
               max_tokens=MAX_TOKENS_DEFAULT, response_format_json=force_json)
    # robust parse (reused logic)
    import re
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    out = None
    try:
        out = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, flags=re.S)
        if m:
            try:
                out = json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return out, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="eval", choices=["eval", "topN", "ids"])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--ids")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    force_json = True
    print(f"LLM backend: {config_summary()}")

    print("Scoring full pool to establish ranks ...")
    candidates = list(load_candidates(ROOT / "data" / "candidates.jsonl"))
    ranked, ref = score_all(candidates)
    cands_by_id = {c["candidate_id"]: c for c in candidates}

    if args.sample == "eval":
        n = args.n or 400
        sample = build_eval_sample(ranked, cands_by_id, n=n)
    elif args.sample == "topN":
        n = args.n or 200
        sample = [r["candidate"] for r in ranked[:n]]
    else:
        n = args.n
        ids = [l.strip() for l in open(args.ids) if l.strip()]
        sample = [cands_by_id[i] for i in ids if i in cands_by_id]

    out_path = Path(args.out)
    done = already_done(out_path)
    todo = [c for c in sample if c["candidate_id"] not in done]
    print(f"Sample: {len(sample)}  already graded: {len(done)}  to grade: {len(todo)}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pacing: NVIDIA free tier trips 429 on sustained calls (tight RPM/TPM
    # window). Pace under the per-minute cap; on 429, wait a FULL 60s window
    # (exponential backoff doesn't align with minute-grain rate limits).
    import os as _os
    min_delay = float(_os.environ.get("LLM_MIN_DELAY_S", "6.0"))
    window_wait = float(_os.environ.get("LLM_429_WAIT_S", "60.0"))

    with open(out_path, "a", encoding="utf-8") as f:
        for k, c in enumerate(todo, 1):
            cid = c["candidate_id"]
            t0 = time.time()
            out, raw = None, None
            # On 429, wait a full rate window and retry — up to 8 windows.
            for attempt in range(8):
                try:
                    out, raw = grade_one(c, None, force_json)
                    break
                except LLMError as e:
                    msg = str(e)
                    if ("429" in msg or "rate" in msg.lower()) and attempt < 7:
                        print(f"  [{cid}] 429, waiting {window_wait:.0f}s for window reset "
                              f"(attempt {attempt+1}/8) ...")
                        time.sleep(window_wait)
                        continue
                    out, raw = None, f"ERROR: {msg}"
                    break
                except Exception as e:
                    out, raw = None, f"ERROR: {type(e).__name__}: {e}"
                    break
            rec = {"candidate_id": cid}
            if isinstance(out, dict):
                rec.update({
                    "grade": out.get("grade"),
                    "fit_score": out.get("fit_score"),
                    "confidence": out.get("confidence"),
                    "reasoning": out.get("reasoning"),
                    "concerns": out.get("concerns"),
                    "rank_when_sampled": next(
                        (i + 1 for i, r in enumerate(ranked)
                         if r["candidate_id"] == cid), None),
                })
            else:
                rec["error"] = (raw or "")[:300]
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if k % 10 == 0 or k == len(todo):
                g = rec.get("grade", "?")
                print(f"  [{k}/{len(todo)}] {cid} grade={g}")
            # pace to respect the rate limit
            elapsed = time.time() - t0
            if elapsed < min_delay:
                time.sleep(min_delay - elapsed)
    print(f"Done. Labels in {out_path}")


if __name__ == "__main__":
    main()
