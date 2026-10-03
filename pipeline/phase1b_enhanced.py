"""Phase 1b offline pipeline: the "Enhanced Model" (GridSearchCV-tuned Random Forest).

A separate, parallel track. It never touches the Thesis Model artifacts. For each of the
8 graph x operator combinations it reuses the SAME embeddings, the SAME stratified 70/30
split (random_state=42) and the SAME 1:1 undersampled training set as the Thesis Model;
only the Random Forest hyperparameters change.

  * GridSearchCV, 5-fold stratified CV (shuffled, random_state=42), scoring='average_precision'
  * fitted on the undersampled training set only
  * the best estimator is evaluated once on the untouched test set

Each finished combination is saved immediately, so an interrupted run resumes where it stopped.

Usage, from the project root (after phase1_thesis embed + classify):
  python -m pipeline.phase1b_enhanced
  python -m pipeline.phase1b_enhanced --only tripartite:hadamard     # a single combination
"""
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from pipeline.common import ARTIFACTS_DIR, BUILD_DIR, GRAPHS, OPERATORS, update_manifest
from pipeline.phase1_thesis import edge_features

PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}
N_CANDIDATES = int(np.prod([len(v) for v in PARAM_GRID.values()]))   # 216


def load_edges(name):
    kv = KeyedVectors.load(f"{ARTIFACTS_DIR}/{name}_wv.kv")
    d = np.load(f"{ARTIFACTS_DIR}/{name}_edges.npz")
    key2idx = {k: i for i, k in enumerate(kv.index_to_key)}
    iu = np.array([key2idx[str(x)] for x in d["u"]])
    iv = np.array([key2idx[str(x)] for x in d["v"]])
    return kv.vectors[iu], kv.vectors[iv], d["y"]


def tune_one(name, method, A, B, y, n_jobs):
    stem = f"{ARTIFACTS_DIR}/{name}_rf_{method}_enhanced"
    result_file = BUILD_DIR / f"enhanced_{name}_{method}.json"
    if result_file.exists():
        print(f"[skip] {name}/{method}: already done ({result_file.name})")
        return json.loads(result_file.read_text())

    X = edge_features(A, B, method)
    # identical to the Thesis Model (pipeline/phase1_thesis.py, A.5)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    rng = np.random.RandomState(42)
    pos = np.where(ytr == 1)[0]
    neg = rng.choice(np.where(ytr == 0)[0], size=len(pos), replace=False)
    sel = np.concatenate([pos, neg])

    t0 = time.time()
    gs = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=1),
        PARAM_GRID,
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=n_jobs,
        refit=True,
        verbose=1,
    ).fit(Xtr[sel], ytr[sel])
    clf = gs.best_estimator_

    prob = clf.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
    p, r, f1, _ = precision_recall_fscore_support(yte, pred, pos_label=1, average="binary",
                                                  zero_division=0)
    row = {"graph": name, "operator": method,
           "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
           "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
           "roc_auc": round(roc_auc_score(yte, prob), 3),
           "pr_auc": round(average_precision_score(yte, prob), 3),
           "baseline_pr_auc": round(yte.mean(), 3),
           "best_params": gs.best_params_,
           "cv_pr_auc": round(gs.best_score_, 3),
           "search_minutes": round((time.time() - t0) / 60, 1)}

    joblib.dump(clf, f"{stem}.joblib", compress=3)
    prec, rec, thr = precision_recall_curve(yte, prob)
    np.savez_compressed(f"{stem}_pr.npz", precision=prec, recall=rec, thresholds=thr)
    result_file.write_text(json.dumps(row, indent=2))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="run a single combination, e.g. tripartite:hadamard")
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    combos = [(g, m) for g in GRAPHS for m in OPERATORS]
    if args.only:
        combos = [tuple(args.only.split(":"))]
    print(f"Grid: {N_CANDIDATES} candidates x 5 folds = {N_CANDIDATES * 5} fits per combination, "
          f"{len(combos)} combination(s)")

    thesis = {(r["graph"], r["operator"]): r for r in
              json.loads((ARTIFACTS_DIR / "manifest.json").read_text())["thesis"]["rerun"]}
    rows, loaded = [], None
    for i, (name, method) in enumerate(combos, 1):
        if loaded != name:
            A, B, y = load_edges(name)
            loaded = name
        print(f"\n=== [{i}/{len(combos)}] {name} / {method} ===", flush=True)
        row = tune_one(name, method, A, B, y, args.n_jobs)
        rows.append(row)
        t = thesis[(name, method)]
        print(f"done {name}/{method} in {row['search_minutes']} min | best {row['best_params']} | "
              f"test PR-AUC {row['pr_auc']:.3f} (thesis model {t['pr_auc']:.3f}) | "
              f"P={row['precision']:.3f} R={row['recall']:.3f}", flush=True)

    # Only publish the section once all eight combinations exist.
    done = [json.loads((BUILD_DIR / f"enhanced_{g}_{m}.json").read_text())
            for g in GRAPHS for m in OPERATORS if (BUILD_DIR / f"enhanced_{g}_{m}.json").exists()]
    if len(done) == len(GRAPHS) * len(OPERATORS):
        update_manifest(["enhanced"], {
            "label": "Enhanced Model (tuned)",
            "classifier": "RandomForestClassifier tuned by GridSearchCV "
                          "(5-fold stratified CV, scoring='average_precision') on the same "
                          "1:1 undersampled training set as the Thesis Model",
            "param_grid": PARAM_GRID,
            "rerun": done,
            "trained": time.strftime("%Y-%m-%d %H:%M"),
        })
        cmp = pd.DataFrame([{"graph": r["graph"], "operator": r["operator"],
                             "thesis_pr_auc": thesis[(r["graph"], r["operator"])]["pr_auc"],
                             "enhanced_pr_auc": r["pr_auc"],
                             "thesis_recall": thesis[(r["graph"], r["operator"])]["recall"],
                             "enhanced_recall": r["recall"]} for r in done])
        print("\nTest-set comparison (same split, same embeddings):")
        print(cmp.to_string(index=False))
    else:
        print(f"\n{len(done)}/8 combinations done; manifest 'enhanced' section is written once all 8 exist.")


if __name__ == "__main__":
    main()
