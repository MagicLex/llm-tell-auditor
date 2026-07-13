"""T-pipeline: train a calibrated human-vs-LLM stylometric classifier.

Two calibrated members, blended:

  * `tells`: scaler+logistic over the 16 stylometric tells from the
    `tell_classifier` feature view. Interpretable: its standardized coefficients
    are the per-tell importances the auditor surfaces.
  * `ngram`: TF-IDF over char 3-5 grams (word-boundary aware) + logistic, on the
    raw section text from `paper_twins`. The classic authorship-attribution
    workhorse; catches habits nobody hand-coded.

The served score is `(1-w)*tells + w*ngram`, with `w` picked on out-of-fold
train predictions. Evidence/attribution in the auditor stays on the tell member
only, by design: char-ngram weights have no honest per-tell story.

Two non-negotiables for this problem:

  * Split by paper_id, never by pair or row. A paper contributes several
    sections (pair_ids); one author's style, or one topic's vocabulary, must
    not straddle train and test. With char-ngrams a pair-level split would let
    the model memorize each paper's vocabulary and lie about AUROC.
  * Scaling/vectorizing lives inside the model (Pipeline), fit on train only.
    No FV MDT here, so there is no train/serve skew and no leakage.

Logistic members (not trees) on purpose: calibrated probabilities give honest
confidence, and the tell member's coefficients stay readable. A gradient-boost
comparison on the tells is printed for the record, never shipped blind.
"""

from __future__ import annotations

import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import hopsworks

# columns in paper_tells that are keys/labels/provenance, not model features
META = {"row_id", "pair_id", "paper_id", "section_idx", "category", "source", "label", "published"}
MODEL_DIR = "tell_classifier_model"
NGRAM_C_GRID = (0.3, 1.0, 3.0)


def _feature_view(fs):
    tells = fs.get_feature_group("paper_tells", version=1)
    feature_cols = [f.name for f in tells.features if f.name not in META]
    # pair_id rides along as a passthrough for the group split; label is the target.
    query = tells.select(feature_cols + ["pair_id", "label"])
    fv = fs.get_or_create_feature_view(
        name="tell_classifier",
        version=1,
        query=query,
        labels=["label"],
        description="Stylometric human(0)/LLM(1) tell features, pair-keyed for group splitting",
    )
    return fv, feature_cols


def _texts(fs) -> pd.DataFrame:
    """pair_id -> (human_text, llm_text), deduped against Hudi dup commits.

    llm_text lives in paper_twins (keyed pair_id); human_text lives in
    arxiv_papers_raw (keyed paper_id+section_idx, which is what pair_id encodes).
    """
    twins = fs.get_feature_group("paper_twins", version=1)
    tw = twins.select(["pair_id", "llm_text"]).read().drop_duplicates(subset=["pair_id"])
    raw = fs.get_feature_group("arxiv_papers_raw", version=1)
    hu = raw.select(["paper_id", "section_idx", "human_text"]).read()
    hu["pair_id"] = hu["paper_id"] + "::" + hu["section_idx"].astype("int64").astype(str)
    hu = hu.drop_duplicates(subset=["pair_id"])[["pair_id", "human_text"]]
    return tw.merge(hu, on="pair_id", how="inner").reset_index(drop=True)


def _tell_pipe() -> Pipeline:
    return Pipeline([("scale", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=2000, C=1.0))])


def _ngram_pipe(C: float) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                  min_df=2, sublinear_tf=True)),
        ("lr", LogisticRegression(max_iter=2000, C=C)),
    ])


def _oof(make, X, y, folds) -> np.ndarray:
    """Out-of-fold P(LLM) from refitting `make()` per fold. Positional indexing."""
    oof = np.zeros(len(y))
    for tr, va in folds:
        m = make()
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return oof


def _metrics(y_true, proba) -> dict:
    pred = (proba >= 0.5).astype(int)
    return {
        "auroc": round(float(roc_auc_score(y_true, proba)), 4),
        "precision": round(float(precision_score(y_true, pred)), 4),
        "recall": round(float(recall_score(y_true, pred)), 4),
        "f1": round(float(f1_score(y_true, pred)), 4),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "brier": round(float(brier_score_loss(y_true, proba)), 4),
    }


def _reliability_plot(y_true, proba, path):
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--", label="perfect")
    plt.plot(mean_pred, frac_pos, "o-", label="tell_classifier (blend)")
    plt.xlabel("mean predicted P(LLM)")
    plt.ylabel("observed fraction LLM")
    plt.title("Calibration (holdout)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _roc_plot(curves, path):
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--")
    for name, y_true, proba, auroc in curves:
        fpr, tpr, _ = roc_curve(y_true, proba)
        plt.plot(fpr, tpr, label=f"{name} AUROC={auroc:.3f}")
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title("ROC (holdout)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _importance_plot(names, coefs, path):
    order = sorted(range(len(coefs)), key=lambda i: coefs[i])
    names_s = [names[i] for i in order]
    coefs_s = [coefs[i] for i in order]
    plt.figure(figsize=(6, 6))
    colors = ["#c44" if c > 0 else "#48a" for c in coefs_s]
    plt.barh(names_s, coefs_s, color=colors)
    plt.axvline(0, color="k", lw=0.8)
    plt.xlabel("standardized coefficient  (>0 -> LLM,  <0 -> human)")
    plt.title("Per-tell importance (tell member)")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main() -> None:
    project = hopsworks.login()
    fs = project.get_feature_store()

    fv, feature_cols = _feature_view(fs)
    X_all, y_all = fv.training_data()
    df = X_all.copy()
    df["label"] = y_all["label"].values if isinstance(y_all, pd.DataFrame) else y_all.values
    # the offline store can return un-compacted duplicate commits for the same row;
    # collapse exact dups so no pair is double-counted (a pair is keyed by pair_id+source,
    # and source is implied by the label, so pair_id+label is one physical row).
    before = len(df)
    df = df.drop_duplicates(subset=["pair_id", "label"]).reset_index(drop=True)
    print(f"deduped {before} -> {len(df)} rows ({before - len(df)} dup rows dropped)", flush=True)

    texts = _texts(fs)
    before = len(df)
    df = df.merge(texts, on="pair_id", how="inner", validate="many_to_one")
    if len(df) != before:
        print(f"WARN: {before - len(df)} tell rows had no twin text and were dropped", flush=True)
    df["text"] = np.where(df["label"] == 1, df["llm_text"], df["human_text"])
    # pair_id is '<paper_id>::<section_idx>'; the split must hold papers out whole
    df["paper_id"] = df["pair_id"].str.split("::").str[0]

    y = df["label"]
    papers = df["paper_id"]
    X_tells = df[feature_cols]
    X_text = df["text"]
    print(f"rows: {len(df)} | pairs: {df['pair_id'].nunique()} | papers: {papers.nunique()} "
          f"| features: {len(feature_cols)}", flush=True)

    # hold whole papers out: paper_id never spans train and test
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(gss.split(X_tells, y, groups=papers))
    y_tr, y_te = y.iloc[tr], y.iloc[te]
    p_tr = papers.iloc[tr]
    print(f"train rows {len(tr)} ({p_tr.nunique()} papers) | test rows {len(te)} "
          f"({papers.iloc[te].nunique()} papers)", flush=True)

    # --- out-of-fold on train (grouped by paper): pick ngram C, blend w, log HGB ---
    folds = list(GroupKFold(n_splits=5).split(tr, y_tr, groups=p_tr))
    Xtr_tells, Xtr_text = X_tells.iloc[tr], X_text.iloc[tr]

    oof_tell = _oof(_tell_pipe, Xtr_tells, y_tr, folds)
    print(f"oof tells        auroc={roc_auc_score(y_tr, oof_tell):.4f}", flush=True)

    best_C, best_auc, oof_ngram = None, -1.0, None
    for C in NGRAM_C_GRID:
        oof = _oof(lambda C=C: _ngram_pipe(C), Xtr_text, y_tr, folds)
        auc = roc_auc_score(y_tr, oof)
        print(f"oof ngram C={C:<4} auroc={auc:.4f}", flush=True)
        if auc > best_auc:
            best_C, best_auc, oof_ngram = C, auc, oof

    oof_hgb = _oof(HistGradientBoostingClassifier, Xtr_tells, y_tr, folds)
    print(f"oof hgb(tells)   auroc={roc_auc_score(y_tr, oof_hgb):.4f}  (comparison only, not shipped)",
          flush=True)

    ws = np.linspace(0.0, 1.0, 21)
    aucs = [roc_auc_score(y_tr, (1 - w) * oof_tell + w * oof_ngram) for w in ws]
    blend_w = float(ws[int(np.argmax(aucs))])
    print(f"blend w={blend_w} (oof auroc={max(aucs):.4f})", flush=True)

    # --- final calibrated members on full train, grouped calibration folds ---
    clf_tell = CalibratedClassifierCV(_tell_pipe(), method="sigmoid", cv=folds)
    clf_tell.fit(Xtr_tells, y_tr)
    clf_ngram = CalibratedClassifierCV(_ngram_pipe(best_C), method="sigmoid", cv=folds)
    clf_ngram.fit(Xtr_text, y_tr)

    pt = clf_tell.predict_proba(X_tells.iloc[te])[:, 1]
    pn = clf_ngram.predict_proba(X_text.iloc[te])[:, 1]
    pb = (1 - blend_w) * pt + blend_w * pn

    m_tell, m_ngram, m_blend = _metrics(y_te, pt), _metrics(y_te, pn), _metrics(y_te, pb)
    for name, m in (("tells", m_tell), ("ngram", m_ngram), ("blend", m_blend)):
        print(f"holdout {name:6s}: {json.dumps(m)}", flush=True)

    metrics = dict(m_blend)
    metrics.update({
        "n_test_pairs": int(df["pair_id"].iloc[te].nunique()),
        "n_test_papers": int(papers.iloc[te].nunique()),
        "auroc_tells": m_tell["auroc"],
        "auroc_ngram": m_ngram["auroc"],
        "blend_w": blend_w,
    })

    # interpretable coefficients from a plain pipeline fit on train (tell importances)
    base = _tell_pipe()
    base.fit(Xtr_tells, y_tr)
    coefs = base.named_steps["lr"].coef_[0].tolist()
    importances = sorted(zip(feature_cols, coefs), key=lambda kv: abs(kv[1]), reverse=True)
    print("top tells:", [f"{n}={c:+.2f}" for n, c in importances[:6]], flush=True)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf_tell, f"{MODEL_DIR}/model.pkl")
    joblib.dump(clf_ngram, f"{MODEL_DIR}/ngram.pkl")
    json.dump({"w": blend_w, "ngram_C": best_C}, open(f"{MODEL_DIR}/blend.json", "w"))
    json.dump(feature_cols, open(f"{MODEL_DIR}/feature_names.json", "w"))
    json.dump({n: round(c, 4) for n, c in importances}, open(f"{MODEL_DIR}/tell_importances.json", "w"))
    _reliability_plot(y_te, pb, f"{MODEL_DIR}/calibration.png")
    _roc_plot([("tells", y_te, pt, m_tell["auroc"]),
               ("ngram", y_te, pn, m_ngram["auroc"]),
               ("blend", y_te, pb, m_blend["auroc"])], f"{MODEL_DIR}/roc.png")
    _importance_plot(feature_cols, coefs, f"{MODEL_DIR}/tell_importance.png")

    mr = project.get_model_registry()
    model = mr.python.create_model(
        name="tell_classifier",
        metrics=metrics,
        description="Blend of two calibrated logistics: 16 stylometric tells + char 3-5 gram "
                    "TF-IDF, human(0)-vs-LLM(1). Trained on content-controlled arXiv rewrite-"
                    "pairs, held out by paper_id (whole papers). Label = LLM-authored (strong "
                    "form), within-provider (Anthropic panel). Attribution comes from the tell "
                    "member only.",
        input_example=Xtr_tells.head(1),
        feature_view=fv,
    )
    model.save(MODEL_DIR)
    print(f"registered tell_classifier v{model.version}: {metrics}", flush=True)


if __name__ == "__main__":
    main()
