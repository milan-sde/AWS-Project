import json
import os

import joblib
import pandas as pd

ONE_HOT_PREFIXES = ("proto_", "service_", "state_")


def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    with open(os.path.join(model_dir, "feature_columns.json")) as f:
        columns = json.load(f)
    return {"model": model, "columns": columns}


def input_fn(request_body, content_type="application/json"):
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    data = json.loads(request_body)
    records = data if isinstance(data, list) else [data]
    return pd.DataFrame(records)


def predict_fn(df, bundle):
    cols = bundle["columns"]
    # one-hot columns may be omitted (treated as 0); every other feature is required
    for c in cols:
        if c not in df.columns and c.startswith(ONE_HOT_PREFIXES):
            df[c] = 0
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing[:5]}")
    X = df[cols].astype(float)
    return bundle["model"].predict_proba(X)[:, 1]


def output_fn(proba, accept="application/json"):
    results = [
        {"prediction": "Attack" if p >= 0.5 else "Normal", "confidence": float(max(p, 1 - p)),
         "attack_probability": float(p)}
        for p in proba
    ]
    return json.dumps(results), "application/json"