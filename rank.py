#!/usr/bin/env python3
"""Single entry point: produce the top-100 submission CSV from candidates.jsonl.

    python rank.py --candidates data/candidates.jsonl --out output/submission.csv

Pure-stdlib, CPU-only, no network — designed to sit inside the challenge's
5-minute / 16 GB ranking budget. (Any embedding precompute happens offline and
is loaded as a cached artifact, not computed here.)
"""
import argparse
import csv
import html
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.loader import load_candidates  # noqa: E402
from src.ranker import score_all  # noqa: E402
from src.reasoning import build_reasoning  # noqa: E402
from src.llm_reranker import apply_rerank, load_grades  # noqa: E402
from src.llm_expansion import load_expansion_grades, apply_expansion  # noqa: E402

TOP_N = 100
HEADER = ["candidate_id", "rank", "score", "reasoning"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/candidates.jsonl")
    ap.add_argument("--out", default="output/submission.csv")
    ap.add_argument("--top", type=int, default=TOP_N)
    ap.add_argument("--rerank-window", type=int, default=200,
                    help="LLM re-ranks the top-N of the ensemble (cached, offline)")
    ap.add_argument("--no-rerank", action="store_true",
                    help="disable LLM re-rank even if a cache is present")
    ap.add_argument("--no-expansion", action="store_true",
                    help="disable the evidence-guided LLM expansion pass "
                         "(on by default if cache/llm_expansion.jsonl exists)")
    args = ap.parse_args()

    t0 = time.time()
    print(f"Loading candidates from {args.candidates} ...")
    candidates = load_candidates_list(args.candidates)
    print(f"  {len(candidates):,} candidates loaded in {time.time()-t0:.1f}s")

    scored, ref = score_all(candidates)
    print(f"  scored {len(scored):,} (honeypots dropped); reference date = {ref}")

    # Optional final stage: LLM re-rank of the top window (cached, no network).
    if not args.no_rerank:
        grades = load_grades()
        if grades:
            before = scored[: args.rerank_window]
            scored = apply_rerank(scored, window=args.rerank_window, grades=grades)
            after = scored[: args.rerank_window]
            moved = sum(1 for a, b in zip(before, after) if a["candidate_id"] != b["candidate_id"])
            print(f"  LLM re-rank applied: {len(grades)} grades cached, "
                  f"{moved}/{len(before)} of top-{args.rerank_window} re-ordered")

    # Evidence-guided expansion promotion (default ON; top-20 frozen, two-gate policy).
    # Promotes manually-audited, high-confidence grade-4 candidates from ranks
    # 201-800 into ranks 21-100. Disable with --no-expansion for the base baseline.
    if not args.no_expansion:
        exp = load_expansion_grades()
        if exp:
            import json as _j
            from src.evidence_graph import evidence_graph_score
            ev_map = {r["candidate_id"]: evidence_graph_score(r["candidate"])["score"]
                      for r in scored}
            base_full = {}
            for line in open(ROOT / "cache" / "llm_rerank.jsonl", encoding="utf-8"):
                try:
                    d = _j.loads(line)
                    if d.get("fit_score") is not None:
                        base_full[d["candidate_id"]] = d
                except _j.JSONDecodeError:
                    continue
            scored, audit = apply_expansion(
                scored, base_full, exp, lambda cid: ev_map.get(cid, 0.0))
            print(f"  Expansion: {len(audit['entered'])} promoted, "
                  f"{len(audit['left'])} displaced (top-20 frozen)")
        else:
            print("  Expansion: no cache found")

    top = scored[: args.top]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [HEADER]
    for rank, row in enumerate(top, start=1):
        reasoning = build_reasoning(row["candidate"], ref, rank)
        rows.append([row["candidate_id"], rank, row["score"], reasoning])

    if out_path.suffix.lower() == ".xlsx":
        write_xlsx(out_path, rows)
    else:
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerows(rows)

    print(f"Wrote {len(top)} rows to {out_path} in {time.time()-t0:.1f}s total")


def load_candidates_list(path):
    return list(load_candidates(path))


def write_xlsx(path: Path, rows: list[list]):
    """Write a minimal XLSX workbook using only the Python stdlib.

    The challenge upload form asks for an Excel spreadsheet, while the official
    validator works on CSV. This keeps the runtime dependency surface unchanged:
    no openpyxl/xlsxwriter needed in the final repo.
    """
    def cell_ref(row_idx: int, col_idx: int) -> str:
        letters = ""
        n = col_idx
        while n:
            n, rem = divmod(n - 1, 26)
            letters = chr(65 + rem) + letters
        return f"{letters}{row_idx}"

    def cell_xml(row_idx: int, col_idx: int, value) -> str:
        ref = cell_ref(row_idx, col_idx)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return f'<c r="{ref}"><v>{value}</v></c>'
        text = html.escape(str(value), quote=False)
        return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'

    sheet_rows = []
    for r_idx, row in enumerate(rows, start=1):
        cells = "".join(cell_xml(r_idx, c_idx, value) for c_idx, value in enumerate(row, start=1))
        sheet_rows.append(f'<row r="{r_idx}">{cells}</row>')

    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<cols><col min="1" max="1" width="18" customWidth="1"/>'
        '<col min="2" max="2" width="8" customWidth="1"/>'
        '<col min="3" max="3" width="12" customWidth="1"/>'
        '<col min="4" max="4" width="110" customWidth="1"/></cols>'
        f'<sheetData>{"".join(sheet_rows)}</sheetData></worksheet>'
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="submission" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>'
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        '</Relationships>'
    )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '</Types>'
    )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types_xml)
        z.writestr("_rels/.rels", rels_xml)
        z.writestr("xl/workbook.xml", workbook_xml)
        z.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        z.writestr("xl/worksheets/sheet1.xml", sheet_xml)


if __name__ == "__main__":
    main()
