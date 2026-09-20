import os
import time
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
)

TRAIN_PATH = "../data/processed/train.csv"
TEST_PATH = "../data/processed/test.csv"
MODELS_DIR = "../models"
REPORTS_DIR = "../reports"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

tr = pd.read_csv(TRAIN_PATH)
te = pd.read_csv(TEST_PATH)

X_tr, y_tr = tr.drop(columns="label").astype(float), tr["label"]
X_te, y_te = te.drop(columns="label").astype(float), te["label"]

models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, eval_metric="logloss", random_state=42),
}

rows = []
for name, model in models.items():
    # training time
    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0

    # prediction latency (average per row over the whole test set)
    t0 = time.time()
    pred = model.predict(X_te)
    latency_ms = (time.time() - t0) / len(X_te) * 1000
    proba = model.predict_proba(X_te)[:, 1]

    cm = confusion_matrix(y_te, pred)
    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_te, pred),
        "Precision": precision_score(y_te, pred),
        "Recall": recall_score(y_te, pred),
        "F1": f1_score(y_te, pred),
        "ROC-AUC": roc_auc_score(y_te, proba),
        "Train time (s)": train_time,
        "Latency (ms/row)": latency_ms,
    }
    rows.append(row)

    print(f"\n{name}")
    print({k: round(v, 4) for k, v in row.items() if k != "Model"})
    print(cm)

    # save confusion matrix plot and model
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Normal", "Attack"]).plot(ax=ax, colorbar=False)
    ax.set_title(f"{name} - confusion matrix")
    fig.tight_layout()
    fig.savefig(f"{REPORTS_DIR}/cm_{name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close(fig)

    joblib.dump(model, f"{MODELS_DIR}/{name.lower().replace(' ', '_')}.joblib")

results = pd.DataFrame(rows).round(4)
results.to_csv(f"{REPORTS_DIR}/model_comparison.csv", index=False)
print("\n", results.to_string(index=False))
print(f"\nSaved results to {REPORTS_DIR}/model_comparison.csv")