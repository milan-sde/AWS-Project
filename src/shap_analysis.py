import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

REPORTS_DIR = "../reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

model = joblib.load("../models/xgboost.joblib")
te = pd.read_csv("../data/processed/test.csv")
X_te = te.drop(columns="label").astype(float)

# SHAP on a random sample keeps it fast (2,000 rows is plenty for the plots)
X = X_te.sample(n=2000, random_state=42).reset_index(drop=True)

explainer = shap.TreeExplainer(model)
sv = explainer(X)          # explanation object, values are in log-odds

# 1. Global importance (mean |SHAP|)
plt.figure()
shap.plots.bar(sv, max_display=15, show=False)
plt.title("Top 15 features by mean |SHAP|")
plt.savefig(f"{REPORTS_DIR}/shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()

# 2. Summary (beeswarm) plot: direction and size of each feature's impact
plt.figure()
shap.plots.beeswarm(sv, max_display=15, show=False)
plt.savefig(f"{REPORTS_DIR}/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

# 3. Individual explanations: one predicted Attack, one predicted Normal
proba = model.predict_proba(X)[:, 1]
for label, idx in [("attack", int(np.argmax(proba))), ("normal", int(np.argmin(proba)))]:
    plt.figure()
    shap.plots.waterfall(sv[idx], max_display=12, show=False)
    plt.savefig(f"{REPORTS_DIR}/shap_waterfall_{label}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved waterfall for {label} sample (P(attack)={proba[idx]:.3f})")

# 4. Save the ranking as a table for your report
importance = pd.DataFrame({
    "feature": X.columns,
    "mean_abs_shap": np.abs(sv.values).mean(axis=0),
}).sort_values("mean_abs_shap", ascending=False)
importance.to_csv(f"{REPORTS_DIR}/shap_importance.csv", index=False)
print(importance.head(15).to_string(index=False))