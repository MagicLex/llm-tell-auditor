"""F-pipeline step 1: ingest arXiv preprints into the `arxiv_papers_raw` feature group.

Same program backfills and runs incrementally: it fetches the most recent papers per
category, parses sections, and upserts by (paper_id, section_idx) so reruns are idempotent.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

import hopsworks
from arxiv_ingest import extract_main_tex, fetch_recent_ids, fetch_source, parse_sections

# diverse spread across 6 fields so the auditor learns LLM style, not one field's house style
CATEGORIES = [
    "cs.LG", "cs.CL", "cs.AI", "cs.CV",           # ML / CS
    "stat.ML", "stat.ME",                          # statistics
    "math.OC", "math.PR",                          # mathematics
    "astro-ph.GA", "cond-mat.stat-mech",           # physics
    "q-bio.NC",                                     # quantitative biology
    "econ.EM",                                      # economics
]
POLITE_DELAY_S = 3  # arXiv asks ~3s between requests

FEATURE_DESCRIPTIONS = {
    "paper_id": "arXiv identifier, e.g. 2607.08754v1",
    "section_idx": "0-based index of the section within the paper",
    "category": "arXiv primary category the paper was fetched from",
    "section_title": "Section heading (Abstract, Introduction, ...)",
    "human_text": "Clean human-authored prose for the section, LaTeX stripped, math/citations removed",
    "word_count": "Number of whitespace-delimited tokens in human_text",
    "published": "arXiv submission timestamp (event time)",
}


def collect(categories: list[str], per_category: int) -> pd.DataFrame:
    rows: list[dict] = []
    for cat in categories:
        papers = fetch_recent_ids(cat, max_results=per_category)
        time.sleep(POLITE_DELAY_S)
        for p in papers:
            blob = fetch_source(p["paper_id"])
            time.sleep(POLITE_DELAY_S)
            if not blob:
                print(f"skip {p['paper_id']}: no source")
                continue
            main_tex = extract_main_tex(blob)
            if not main_tex:
                print(f"skip {p['paper_id']}: no main tex")
                continue
            sections = parse_sections(main_tex)
            for idx, (title, prose) in enumerate(sections):
                rows.append(
                    {
                        "paper_id": p["paper_id"],
                        "section_idx": idx,
                        "category": p["category"],
                        "section_title": title[:200],
                        "human_text": prose,
                        "word_count": len(prose.split()),
                        "published": p["published"],
                    }
                )
            print(f"ok {p['paper_id']}: {len(sections)} sections")
    df = pd.DataFrame(rows)
    if not df.empty:
        df["published"] = pd.to_datetime(df["published"], utc=True)
    return df


def write_fg(df: pd.DataFrame) -> None:
    project = hopsworks.login()
    fs = project.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name="arxiv_papers_raw",
        version=1,
        description="Human-authored arXiv preprint sections (clean prose) for the LLM tell auditor",
        primary_key=["paper_id", "section_idx"],
        event_time="published",
        online_enabled=False,
    )
    fg.insert(df)
    for name, desc in FEATURE_DESCRIPTIONS.items():
        try:
            fg.update_feature_description(name, desc)
        except Exception as e:
            print(f"desc {name}: {e}")
    print(f"wrote {len(df)} section rows for {df['paper_id'].nunique()} papers")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=5)
    ap.add_argument("--categories", nargs="+", default=CATEGORIES)
    ap.add_argument("--dry-run", action="store_true", help="collect only, do not write FG")
    args = ap.parse_args()

    df = collect(args.categories, args.per_category)
    if df.empty:
        print("no rows collected")
        return
    print(df.groupby("category")["paper_id"].nunique())
    if args.dry_run:
        print(df[["paper_id", "section_title", "word_count"]].head(20).to_string())
        return
    write_fg(df)


if __name__ == "__main__":
    main()
