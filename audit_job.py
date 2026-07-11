"""I-pipeline job: audit recent arXiv papers, write dossiers to `paper_dossiers`.

For each paper it fetches the LaTeX source, scores every section with the pinned
`tell_classifier` v1, and stores a dossier: paper-level metrics (typed columns,
so tell drift is monitorable, the arms-race signal from the spec) plus the full
per-section evidence as a JSON string the app renders server-side.

Runs as a Hopsworks PYTHON job, or in the terminal (deps already present). arXiv
asks ~3s between requests, so papers are audited sequentially with a polite gap.
Idempotent: upsert by paper_id, and `--skip-existing` avoids re-fetching papers
already in the store.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone

import pandas as pd

import hopsworks

CODE_DIR = "/hopsfs/Users/meb10000/010_llm_tell_auditor"
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from arxiv_ingest import fetch_recent_ids  # noqa: E402
from auditor import audit_paper, load_auditor  # noqa: E402

# the four categories the spec targets
CATEGORIES = ["cs.LG", "cs.CL", "stat.ML", "cs.AI"]

FEATURE_DESCRIPTIONS = {
    "paper_id": "arXiv identifier (primary key)",
    "title": "Paper title from arXiv metadata",
    "category": "arXiv primary category the paper was sampled from",
    "family": "Tell family audited (stylometric_polish; the only family F built)",
    "model_version": "tell_classifier model version used to score",
    "n_sections": "Sections scored (>= 40 words, the training-distribution floor)",
    "n_flagged": "Sections with P(LLM) >= 0.5",
    "flagged_share": "n_flagged / n_sections; the paper-level tell rate, watch for drift",
    "mean_proba": "Mean P(LLM) across scored sections",
    "max_proba": "Max P(LLM) across scored sections",
    "sections_json": "Full per-section evidence (tells fired, values, excerpts) for the app",
    "audited_at": "When this dossier was produced (event time)",
}


def _dossiers_fg(fs):
    return fs.get_or_create_feature_group(
        name="paper_dossiers",
        version=1,
        description="Per-paper LLM-tell dossiers from the tell auditor: paper-level "
                    "metrics plus full per-section evidence. Signal, not verdict.",
        primary_key=["paper_id"],
        event_time="audited_at",
        online_enabled=False,
    )


def _existing_ids(fs) -> set[str]:
    try:
        df = fs.get_feature_group("paper_dossiers", version=1).read()
        return set(df["paper_id"].tolist()) if not df.empty else set()
    except Exception:
        return set()


def _row(dossier: dict, category: str, now: datetime) -> dict:
    return {
        "paper_id": dossier["paper_id"],
        "title": dossier["title"][:500],
        "category": category,
        "family": dossier["family"],
        "model_version": int(dossier["model_version"]),
        "n_sections": int(dossier["n_sections"]),
        "n_flagged": int(dossier["n_flagged"]),
        "flagged_share": float(dossier["flagged_share"]),
        "mean_proba": float(dossier["mean_proba"]),
        "max_proba": float(dossier["max_proba"]),
        "sections_json": json.dumps({"top_tells": dossier["top_tells"],
                                     "sections": dossier["sections"]}),
        "audited_at": now,
    }


def _flush(fg, rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["audited_at"] = pd.to_datetime(df["audited_at"], utc=True)
    fg.insert(df)
    for name, desc in FEATURE_DESCRIPTIONS.items():
        try:
            fg.update_feature_description(name, desc)
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=8, help="recent papers to pull per category")
    ap.add_argument("--categories", nargs="+", default=CATEGORIES)
    ap.add_argument("--skip-existing", action="store_true", help="skip papers already in paper_dossiers")
    ap.add_argument("--gap", type=float, default=3.0, help="seconds between arXiv fetches (rate limit)")
    ap.add_argument("--flush-every", type=int, default=10)
    args = ap.parse_args()

    project = hopsworks.login()
    fs = project.get_feature_store()

    mr = project.get_model_registry()
    model_dir = mr.get_model("tell_classifier", version=1).download()
    auditor = load_auditor(model_dir)
    print(f"loaded auditor from {model_dir}", flush=True)

    # gather candidates: recent papers per category, deduped, id -> (title, category)
    candidates: dict[str, tuple[str, str]] = {}
    for cat in args.categories:
        try:
            for p in fetch_recent_ids(cat, max_results=args.per_category):
                candidates.setdefault(p["paper_id"], (p["title"], cat))
        except Exception as e:
            print(f"list {cat} failed: {str(e)[:120]}", flush=True)
        time.sleep(args.gap)

    skip = _existing_ids(fs) if args.skip_existing else set()
    todo = [(pid, t, c) for pid, (t, c) in candidates.items() if pid not in skip]
    print(f"candidates: {len(candidates)} | skipping existing: {len(skip)} | to audit: {len(todo)}", flush=True)

    fg = _dossiers_fg(fs)
    buffer: list[dict] = []
    ok = fail = 0
    for pid, title, cat in todo:
        try:
            dossier = audit_paper(pid, auditor, title=title)
            if dossier is None:
                fail += 1
                print(f"skip {pid}: no source / no scorable section", flush=True)
            else:
                ok += 1
                buffer.append(_row(dossier, cat, datetime.now(timezone.utc)))
                print(f"ok {pid} [{cat}] sections={dossier['n_sections']} "
                      f"flagged={dossier['n_flagged']} mean={dossier['mean_proba']} "
                      f"max={dossier['max_proba']}", flush=True)
        except Exception as e:
            fail += 1
            print(f"FAIL {pid}: {str(e)[:160]}", flush=True)
        if len(buffer) >= args.flush_every:
            _flush(fg, buffer)
            print(f"  flushed {len(buffer)}", flush=True)
            buffer = []
        time.sleep(args.gap)

    _flush(fg, buffer)
    print(f"done: {ok} audited, {fail} skipped/failed", flush=True)


if __name__ == "__main__":
    main()
