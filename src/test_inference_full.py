# src/test_inference_full.py  (run from the project root)
import sys
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score
sys.path.insert(0, "src")
import inference

bundle = inference.model_fn("models/local_out")
te = pd.read_csv("data/processed/test.csv")

proba = inference.predict_fn(te.drop(columns="label").copy(), bundle)
pred = (proba >= 0.5).astype(int)
print("accuracy:", accuracy_score(te["label"], pred))   # expect 0.8762
print("recall:  ", recall_score(te["label"], pred))     # expect 0.9867