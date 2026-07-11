"""T-pipeline: train a calibrated human-vs-LLM stylometric classifier.

Consumes the `tell_classifier` feature view (built here from `paper_tells`) and
registers a calibrated logistic model. Two non-negotiables for this problem:

  * Split by pair_id, never by row. Each pair is one human section + its content-
    controlled twin sharing a pair_id; a random row split would put a paper on
    both sides and leak style. GroupShuffleSplit on pair_id holds papers out whole.
  * Scaling lives inside the model (Pipeline), fit on train only. No FV MDT here,
    so there is no train/serve skew and no leakage from a scaler fit on test rows.

Logistic (not a tree) on purpose: standardized coefficients are the per-tell
importances the auditor surfaces, and calibrated probabilities give honest
confidence instead of a bare verdict.
"""

from __future__ import annotations

import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
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


def _reliability_plot(y_true, proba, path):
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--", label="perfect")
    plt.plot(mean_pred, frac_pos, "o-", label="tell_classifier")
    plt.xlabel("mean predicted P(LLM)")
    plt.ylabel("observed fraction LLM")
    plt.title("Calibration (holdout)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _roc_plot(y_true, proba, auroc, path):
    fpr, tpr, _ = roc_curve(y_true, proba)
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(fpr, tpr, label=f"AUROC={auroc:.3f}")
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
    plt.title("Per-tell importance")
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
    y = df["label"]
    groups = df["pair_id"]
    X = df[feature_cols]
    print(f"rows: {len(X)} | pairs: {groups.nunique()} | features: {len(feature_cols)}", flush=True)

    # hold whole papers out: pair_id never spans train and test
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(gss.split(X, y, groups=groups))
    X_tr, X_te = X.iloc[tr], X.iloc[te]
    y_tr, y_te = y.iloc[tr], y.iloc[te]
    g_tr = groups.iloc[tr]
    print(f"train rows {len(X_tr)} ({g_tr.nunique()} pairs) | test rows {len(X_te)} "
          f"({groups.iloc[te].nunique()} pairs)", flush=True)

    base = Pipeline([("scale", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=2000, C=1.0))])
    # calibrate on group-respecting folds so a pair never straddles fit/calibrate
    cv = list(GroupKFold(n_splits=5).split(X_tr, y_tr, groups=g_tr))
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=cv)
    clf.fit(X_tr, y_tr)

    proba = clf.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "auroc": round(float(roc_auc_score(y_te, proba)), 4),
        "precision": round(float(precision_score(y_te, pred)), 4),
        "recall": round(float(recall_score(y_te, pred)), 4),
        "f1": round(float(f1_score(y_te, pred)), 4),
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "brier": round(float(brier_score_loss(y_te, proba)), 4),
        "n_test_pairs": int(groups.iloc[te].nunique()),
    }
    print("holdout metrics:", json.dumps(metrics), flush=True)

    # interpretable coefficients from a plain pipeline fit on train (tell importances)
    base.fit(X_tr, y_tr)
    coefs = base.named_steps["lr"].coef_[0].tolist()
    importances = sorted(zip(feature_cols, coefs), key=lambda kv: abs(kv[1]), reverse=True)
    print("top tells:", [f"{n}={c:+.2f}" for n, c in importances[:6]], flush=True)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, f"{MODEL_DIR}/model.pkl")
    json.dump(feature_cols, open(f"{MODEL_DIR}/feature_names.json", "w"))
    json.dump({n: round(c, 4) for n, c in importances}, open(f"{MODEL_DIR}/tell_importances.json", "w"))
    _reliability_plot(y_te, proba, f"{MODEL_DIR}/calibration.png")
    _roc_plot(y_te, proba, metrics["auroc"], f"{MODEL_DIR}/roc.png")
    _importance_plot(feature_cols, coefs, f"{MODEL_DIR}/tell_importance.png")

    mr = project.get_model_registry()
    model = mr.python.create_model(
        name="tell_classifier",
        metrics=metrics,
        description="Calibrated logistic human(0)-vs-LLM(1) classifier over 16 stylometric "
                    "tells. Trained on content-controlled arXiv rewrite-pairs, held out by "
                    "pair_id. Label = LLM-authored (strong form), within-provider (Anthropic panel).",
        input_example=X_tr.head(1),
        feature_view=fv,
    )
    model.save(MODEL_DIR)
    print(f"registered tell_classifier: {metrics}", flush=True)


if __name__ == "__main__":
    main()
