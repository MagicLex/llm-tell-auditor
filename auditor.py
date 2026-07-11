"""I-pipeline deep module: audit an arXiv paper for LLM writing tells.

Given an arXiv id, fetch and parse the LaTeX source (same `parse_sections` the F
pipeline used, so no train/serve skew), compute the 16 stylometric tells per
section (same `tell_features.features`), and score each section with the pinned
`tell_classifier` model. The output is an evidence dossier: per section, the
calibrated P(LLM) and which tells pushed it there, plus paper-level counts.

Honesty rules baked in:
  * One tell family only: stylometric polish tells. The tortured-phrase family
    from the spec was never built in F, so this auditor never claims it.
  * Signal, not verdict. We report which known LLM writing tells fired, never
    "real vs BS". A section is "flagged" (P >= 0.5), never "AI-written".
  * The score is the model's calibrated probability. The per-tell contributions
    are an explanation derived from the model's own averaged linear parts, so
    they are consistent with what actually predicts.

Pure functions take text; `audit_paper` wraps them with the network fetch.
"""

from __future__ import annotations

import math
import os
import re

import joblib
import numpy as np
import pandas as pd

import arxiv_ingest
from tell_features import FEATURE_DOC, features

# match the F1 training-data floor: the classifier only ever saw sections of at
# least this many words, so scoring shorter ones would be out of distribution.
MIN_WORDS = 40
FLAG_THRESHOLD = 0.5
FAMILY = "stylometric_polish"

# a bare arXiv id (2607.08754 / 2607.08754v1) or an arxiv.org/abs/... URL
_ARXIV_ID = re.compile(r"\b(\d{4}\.\d{4,5}(?:v\d+)?)\b")


def extract_arxiv_id(s: str) -> str | None:
    """Return an arXiv id if the input is (or contains, for a URL) one, else None."""
    s = s.strip()
    m = _ARXIV_ID.search(s)
    if m and (m.group(0) == s or "arxiv.org" in s.lower()):
        return m.group(1)
    return None


def load_auditor(model_dir: str) -> dict:
    """Load the calibrated classifier plus an averaged linear model for explanation.

    The calibrated model is an ensemble of `n_folds` scaler+logistic pipelines.
    We keep the whole ensemble for the honest calibrated probability, and average
    the folds' scaler stats and coefficients into one linear model to attribute a
    section's score to individual tells.
    """
    clf = joblib.load(os.path.join(model_dir, "model.pkl"))
    feature_names = _read_json(os.path.join(model_dir, "feature_names.json"))

    means, scales, coefs, intercepts = [], [], [], []
    for cc in clf.calibrated_classifiers_:
        est = getattr(cc, "estimator", None) or getattr(cc, "base_estimator")
        sc, lr = est.named_steps["scale"], est.named_steps["lr"]
        means.append(sc.mean_)
        scales.append(sc.scale_)
        coefs.append(lr.coef_[0])
        intercepts.append(lr.intercept_[0])

    return {
        "clf": clf,
        "feature_names": feature_names,
        "mean": np.mean(means, axis=0),
        "scale": np.mean(scales, axis=0),
        "coef": np.mean(coefs, axis=0),
        "intercept": float(np.mean(intercepts)),
    }


def _read_json(path: str):
    import json
    with open(path) as f:
        return json.load(f)


def _row(feat: dict, feature_names: list[str]) -> pd.DataFrame:
    # a named single-row frame: the model was fit with feature names, so match it
    return pd.DataFrame([[feat[n] for n in feature_names]], columns=feature_names, dtype=float)


def score_section(text: str, auditor: dict) -> dict:
    """Score one section: calibrated P(LLM) plus per-tell contributions.

    Contribution of tell i to the log-odds is coef_i * z_i, where z_i is the
    section's value standardized by the training mean/scale. Positive pushes
    toward LLM, negative toward human. These sum (with the intercept) to the
    linear log-odds; the reported probability is the calibrated one.
    """
    feat = features(text)
    names = auditor["feature_names"]
    x = _row(feat, names)
    proba = float(auditor["clf"].predict_proba(x)[0, 1])

    xv = x.iloc[0].to_numpy()
    z = (xv - auditor["mean"]) / auditor["scale"]
    contrib = auditor["coef"] * z
    tells = [
        {
            "tell": names[i],
            "doc": FEATURE_DOC[names[i]],
            "value": round(float(xv[i]), 4),
            "z": round(float(z[i]), 3),
            "contribution": round(float(contrib[i]), 3),
        }
        for i in range(len(names))
    ]
    tells.sort(key=lambda t: t["contribution"], reverse=True)
    return {"proba": round(proba, 4), "features": feat, "tells": tells}


# the 5 token-level tells map to a highlight category; the other 11 are
# distributional and have no single word to mark.
_HL_FEATURE = {"transition_rate": "transition", "booster_rate": "booster",
               "hedge_rate": "hedge", "dash_rate": "dash", "semicolon_rate": "punc"}


def _hl_levels(tells: list[dict]) -> dict:
    """Per-category highlight strength (1=underline, 2=wash, 3=marker) from how
    much each token-level tell moved this passage's score."""
    lv = {}
    for t in tells:
        cat = _HL_FEATURE.get(t["tell"])
        if cat:
            m = abs(t["contribution"])
            lv[cat] = 3 if m > 0.5 else (2 if m > 0.15 else 1)
    return lv


def score_item(title: str, prose: str, auditor: dict) -> dict:
    """Score one titled unit of prose into the dossier section shape."""
    scored = score_section(prose, auditor)
    fired = [t for t in scored["tells"] if t["contribution"] > 0]
    return {
        "title": title,
        "n_words": len(prose.split()),
        "proba": scored["proba"],
        "flagged": scored["proba"] >= FLAG_THRESHOLD,
        "top_tells": fired[:5],
        "hl_levels": _hl_levels(scored["tells"]),
        "excerpt": prose[:320],
    }


def iter_paper_sections(arxiv_id: str, auditor: dict):
    """Yield scored section dicts one at a time (for streaming). Silent if the
    source cannot be fetched; skips sections below the training-word floor."""
    blob = arxiv_ingest.fetch_source(arxiv_id)
    if not blob:
        return
    main_tex = arxiv_ingest.extract_main_tex(blob)
    if not main_tex:
        return
    for sec_title, prose in arxiv_ingest.parse_sections(main_tex):
        if len(prose.split()) >= MIN_WORDS:  # in the training distribution
            yield score_item(sec_title, prose, auditor)


def split_paragraphs(text: str, min_words: int = 25) -> list[str]:
    """Split pasted prose into scorable items: paragraphs on blank lines, or, for
    a single blob, groups of ~sentences. Short fragments merge into the previous
    item so no item is too small to carry a stylometric signal."""
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(parts) <= 1:
        sents = re.split(r"(?<=[.!?])\s+", text.strip())
        parts, cur = [], ""
        for s in sents:
            cur = (cur + " " + s).strip()
            if len(cur.split()) >= 55:
                parts.append(cur)
                cur = ""
        if cur:
            parts.append(cur)
    merged: list[str] = []
    for p in parts or [text]:
        if merged and len(p.split()) < min_words:
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    return merged


def audit_paper(arxiv_id: str, auditor: dict, title: str | None = None) -> dict | None:
    """Fetch, parse, and audit a paper. Returns a dossier dict, or None if the
    source could not be fetched or held no scorable section."""
    sections = list(iter_paper_sections(arxiv_id, auditor))
    if not sections:
        return None

    probas = [s["proba"] for s in sections]
    flagged = [s for s in sections if s["flagged"]]
    # paper-level tell drive: sum each tell's contribution across flagged sections
    drive: dict[str, float] = {}
    for s in flagged:
        for t in s["top_tells"]:
            drive[t["tell"]] = drive.get(t["tell"], 0.0) + t["contribution"]
    top_paper_tells = sorted(drive.items(), key=lambda kv: kv[1], reverse=True)[:5]

    return {
        "paper_id": arxiv_id,
        "title": title or arxiv_id,
        "family": FAMILY,
        "model_version": 1,
        "n_sections": len(sections),
        "n_flagged": len(flagged),
        "flagged_share": round(len(flagged) / len(sections), 4),
        "mean_proba": round(float(np.mean(probas)), 4),
        "max_proba": round(float(max(probas)), 4),
        "top_tells": [{"tell": t, "doc": FEATURE_DOC[t], "drive": round(d, 3)}
                      for t, d in top_paper_tells],
        "sections": sections,
    }


def audit_text(text: str, auditor: dict) -> dict:
    """Score an arbitrary passage (not a paper) as one unit. Returns the same
    tell shape as a section, plus a `short` flag when the passage is below the
    training-distribution floor so the caller can warn the score is unreliable."""
    scored = score_section(text, auditor)
    n_words = len(text.split())
    fired = [t for t in scored["tells"] if t["contribution"] > 0]
    return {
        "kind": "text",
        "proba": scored["proba"],
        "flagged": scored["proba"] >= FLAG_THRESHOLD,
        "short": n_words < MIN_WORDS,
        "n_words": n_words,
        "top_tells": fired[:6],
        "all_tells": scored["tells"],
        "excerpt": text[:400],
    }


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


if __name__ == "__main__":
    import json
    import sys

    mdir = os.environ.get("MODEL_DIR")
    if not mdir:
        import hopsworks
        m = hopsworks.login().get_model_registry().get_model("tell_classifier", version=1)
        mdir = m.download()
    aud = load_auditor(mdir)
    ident = sys.argv[1] if len(sys.argv) > 1 else "2607.08754"
    dossier = audit_paper(ident, aud)
    if not dossier:
        print(f"no dossier for {ident}")
        sys.exit(1)
    # print a compact summary, full sections as JSON
    print(f"\n=== {dossier['paper_id']}  {dossier['title']}")
    print(f"family={dossier['family']}  sections={dossier['n_sections']}  "
          f"flagged={dossier['n_flagged']} ({dossier['flagged_share']:.0%})  "
          f"mean P(LLM)={dossier['mean_proba']}  max={dossier['max_proba']}")
    print("paper-level tells fired:")
    for t in dossier["top_tells"]:
        print(f"  {t['tell']:16s} drive {t['drive']:+.2f}  {t['doc']}")
    print("\nper-section:")
    for s in dossier["sections"]:
        flag = "FLAG" if s["flagged"] else "    "
        top = ", ".join(f"{t['tell']}{t['contribution']:+.2f}" for t in s["top_tells"][:3])
        print(f"  [{flag}] P={s['proba']:.3f}  {s['title'][:40]:40s}  {top}")
