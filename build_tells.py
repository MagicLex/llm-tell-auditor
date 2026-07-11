"""F-pipeline step 3 runner: compute stylometric tell features into `paper_tells`.

For every matched pair in `paper_twins`, emit two rows: the human section
(label 0) and the LLM twin (label 1), each with the same MIT feature vector.
Content-controlled by pair_id, so training splits by pair (never by row) to
avoid leaking a paper across train/test.

No claude here, so this is an ordinary job (run in the terminal now, schedule
later). Idempotent and resumable by (pair_id, source).
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

import hopsworks

# the deployed entry is a frozen upload; put the live code dir on the path so the
# job imports the same tell_features.py we validated, not a duplicate.
CODE_DIR = "/hopsfs/Users/meb10000/010_llm_tell_auditor"
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from tell_features import FEATURE_DOC, FEATURE_NAMES, features  # noqa: E402

BASE_DOC = {
    "row_id": "Unique row key, '<pair_id>::<source>'",
    "pair_id": "Content-controlled pair key; ties a human section to its LLM twin",
    "paper_id": "arXiv identifier",
    "section_idx": "0-based section index",
    "category": "arXiv primary category",
    "source": "'human' or 'llm'",
    "label": "1 if the text is the LLM twin, 0 if the human original",
    "published": "arXiv submission timestamp (event time)",
}


def _existing_row_ids(fs) -> set[str]:
    try:
        df = fs.get_feature_group("paper_tells", version=1).read()
        return set(df["row_id"].tolist()) if not df.empty else set()
    except Exception:
        return set()


def _fg(fs):
    return fs.get_or_create_feature_group(
        name="paper_tells",
        version=1,
        description="Stylometric tell features for human vs LLM-twin arXiv sections (labelled, pair-keyed)",
        primary_key=["row_id"],
        event_time="published",
        online_enabled=False,
    )


def _row(meta: dict, text: str, source: str, label: int) -> dict:
    row = {
        "row_id": f"{meta['pair_id']}::{source}",
        "pair_id": meta["pair_id"],
        "paper_id": meta["paper_id"],
        "section_idx": int(meta["section_idx"]),
        "category": meta["category"],
        "source": source,
        "label": label,
        "published": meta["published"],
    }
    row.update({k: float(v) for k, v in features(text).items()})
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max pairs this run (0 = all)")
    args = ap.parse_args()

    project = hopsworks.login()
    fs = project.get_feature_store()

    twins = fs.get_feature_group("paper_twins", version=1).read()
    raw = fs.get_feature_group("arxiv_papers_raw", version=1).read()
    raw["pair_id"] = raw["paper_id"] + "::" + raw["section_idx"].astype(str)
    human_by_pair = raw.set_index("pair_id")["human_text"].to_dict()

    done = _existing_row_ids(fs)
    fg = _fg(fs)

    pairs = twins[~twins["pair_id"].isin({r.rsplit("::", 1)[0] for r in done})]
    if args.limit:
        pairs = pairs.head(args.limit)
    print(f"twins: {len(twins)} | rows already built: {len(done)} | pairs pending: {len(pairs)}")

    rows: list[dict] = []
    for _, t in pairs.iterrows():
        human = human_by_pair.get(t["pair_id"])
        if not human:
            print(f"skip {t['pair_id']}: no matching human section")
            continue
        meta = {
            "pair_id": t["pair_id"],
            "paper_id": t["paper_id"],
            "section_idx": t["section_idx"],
            "category": t["category"],
            "published": t["published"],
        }
        rows.append(_row(meta, human, "human", 0))
        rows.append(_row(meta, t["llm_text"], "llm", 1))

    if not rows:
        print("nothing to build")
        return

    fg.insert(pd.DataFrame(rows))
    for name, desc in {**BASE_DOC, **FEATURE_DOC}.items():
        try:
            fg.update_feature_description(name, desc)
        except Exception:
            pass
    print(f"wrote {len(rows)} feature rows ({len(rows) // 2} pairs), {len(FEATURE_NAMES)} features each")


if __name__ == "__main__":
    main()
