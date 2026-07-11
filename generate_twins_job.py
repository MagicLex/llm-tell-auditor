"""F-pipeline step 2 (SOTA job): generate LLM rewrite-twins via the Anthropic API.

Runs as a Hopsworks PYTHON job in the `tell-auditor` env (which has `anthropic`),
reads ANTHROPIC_API_KEY from the project secret, and generates twins in parallel
through the firewall with a within-provider writer PANEL (opus/sonnet/haiku).
Idempotent and resumable by pair_id, so reruns skip finished pairs.

This supersedes the in-pod claude-CLI backfill: the claude runtime was terminal-
pod-only (BLOCKERS #010), but the API path runs anywhere the key reaches, so it
becomes an ordinary schedulable, parallel job.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import hopsworks

# the deployed entry is a frozen upload; put the live code dir on the path so the
# job imports the same firewall_api.py we validated, not a duplicate.
CODE_DIR = "/hopsfs/Users/meb10000/010_llm_tell_auditor"
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import anthropic  # noqa: E402
from firewall_api import pick_writer, read_skeleton, write_twin  # noqa: E402

FEATURE_DESCRIPTIONS = {
    "pair_id": "Unique pair key, '<paper_id>::<section_idx>', ties a twin to its human original",
    "paper_id": "arXiv identifier",
    "section_idx": "0-based section index within the paper",
    "category": "arXiv primary category",
    "section_title": "Section heading",
    "writer_model": "Panel model that authored the twin (opus/sonnet/haiku)",
    "n_claims": "Number of claims in the extracted content skeleton (content-completeness proxy)",
    "llm_text": "LLM-authored twin: same content as the human section, written from a prose-free skeleton",
    "human_word_count": "Word count of the human original section",
    "twin_word_count": "Word count of the LLM twin",
    "published": "arXiv submission timestamp (event time), carried from the raw section",
}


def _existing_pairs(fs) -> set[str]:
    try:
        df = fs.get_feature_group("paper_twins", version=1).read()
        return set(df["pair_id"].tolist()) if not df.empty else set()
    except Exception:
        return set()


def _twins_fg(fs):
    return fs.get_or_create_feature_group(
        name="paper_twins",
        version=1,
        description="LLM rewrite-twins of arXiv sections (content-controlled), for the tell auditor",
        primary_key=["pair_id"],
        event_time="published",
        online_enabled=False,
    )


def _make_twin(row: dict, client: anthropic.Anthropic) -> dict | None:
    try:
        skel = read_skeleton(row["human_text"], client)
        model = pick_writer(row["pair_id"])
        twin = write_twin(skel, model, client)
        if len(twin.split()) < 20:
            raise ValueError("twin too short")
        return {
            "pair_id": row["pair_id"],
            "paper_id": row["paper_id"],
            "section_idx": int(row["section_idx"]),
            "category": row["category"],
            "section_title": row["section_title"],
            "writer_model": model,
            "n_claims": len(skel.get("claims", [])),
            "llm_text": twin,
            "human_word_count": int(row["word_count"]),
            "twin_word_count": len(twin.split()),
            "published": row["published"],
        }
    except Exception as e:
        print(f"FAIL {row['pair_id']}: {str(e)[:160]}", flush=True)
        return None


def _flush(fg, rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["published"] = pd.to_datetime(df["published"], utc=True)
    fg.insert(df)
    for name, desc in FEATURE_DESCRIPTIONS.items():
        try:
            fg.update_feature_description(name, desc)
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max sections this run (0 = all pending)")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--flush-every", type=int, default=20)
    ap.add_argument("--min-words", type=int, default=40)
    args = ap.parse_args()

    project = hopsworks.login()
    fs = project.get_feature_store()
    os.environ["ANTHROPIC_API_KEY"] = hopsworks.get_secrets_api().get_secret("ANTHROPIC_API_KEY").value
    client = anthropic.Anthropic()

    raw = fs.get_feature_group("arxiv_papers_raw", version=1).read()
    done = _existing_pairs(fs)
    twins_fg = _twins_fg(fs)

    raw = raw[raw["word_count"] >= args.min_words].copy()
    raw["pair_id"] = raw["paper_id"] + "::" + raw["section_idx"].astype(str)
    pending = raw[~raw["pair_id"].isin(done)].sort_values(["paper_id", "section_idx"])
    if args.limit:
        pending = pending.head(args.limit)
    rows = pending.to_dict("records")
    print(f"raw sections: {len(raw)} | already twinned: {len(done)} | pending this run: {len(rows)}", flush=True)

    buffer: list[dict] = []
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(_make_twin, r, client): r["pair_id"] for r in rows}
        for fut in as_completed(futures):
            res = fut.result()
            if res is None:
                fail += 1
            else:
                ok += 1
                buffer.append(res)
                print(f"ok {res['pair_id']} [{res['writer_model']}] claims={res['n_claims']}", flush=True)
            if len(buffer) >= args.flush_every:
                _flush(twins_fg, buffer)
                print(f"  flushed {len(buffer)} ({ok} done, {fail} failed)", flush=True)
                buffer = []

    _flush(twins_fg, buffer)
    print(f"done: {ok} twinned, {fail} failed", flush=True)


if __name__ == "__main__":
    main()
