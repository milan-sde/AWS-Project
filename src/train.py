import argparse
import json
import os
import time

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="xgboost")   # xgboost | random_forest | decision_tree
    p.add_argument("--n_estimators", type=int, default=200)
    p.add_argument("--max_depth", type=int, default=6)
    p.add_argument("--learning_rate", type=float, default=0.1)
    p.add_argument("--train_dir", default=os.environ.get("SM_CHANNEL_TRAIN", "data/processed"))
    p.add_argument("--test_dir", default=os.environ.get("SM_CHANNEL_TEST", "data/processed"))
    p.add_argument("--model_dir", default=os.environ.get("SM_MODEL_DIR", "models/local_out"))
    args = p.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    tr = pd.read_csv(os.path.join(args.train_dir, "train.csv"))
    te = pd.read_csv(os.path.join(args.test_dir, "test.csv"))
    X_tr, y_tr = tr.drop(columns="label").astype(float), tr["label"]
    X_te, y_te = te.drop(columns="label").astype(float), te["label"]
    print(f"train={X_tr.shape} test={X_te.shape}")

    if args.model == "xgboost":
        model = XGBClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth,
                              learning_rate=args.learning_rate, eval_metric="logloss",
                              random_state=42)
    elif args.model == "random_forest":
        model = RandomForestClassifier(n_estimators=args.n_estimators, n_jobs=-1, random_state=42)
    else:
        model = DecisionTreeClassifier(max_depth=args.max_depth, random_state=42)

    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0

    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]
    metrics = {
        "model": args.model,
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred),
        "recall": recall_score(y_te, pred),
        "f1": f1_score(y_te, pred),
        "roc_auc": roc_auc_score(y_te, proba),
        "train_time_s": train_time,
    }
    # printed lines go to CloudWatch Logs when run on SageMaker
    for k, v in metrics.items():
        print(f"{k}: {v}")

    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
    with open(os.path.join(args.model_dir, "feature_columns.json"), "w") as f:
        json.dump(list(X_tr.columns), f)
    with open(os.path.join(args.model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)