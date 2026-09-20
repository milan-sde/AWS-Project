import json
import pandas as pd
import sys
sys.path.insert(0, "src")
import inference

bundle = inference.model_fn("models/local_out")   # written by python3 src/train.py

te = pd.read_csv("data/processed/test.csv")
sample = te.drop(columns="label").head(3)
body = sample.to_json(orient="records")

df = inference.input_fn(body)
out, _ = inference.output_fn(inference.predict_fn(df, bundle))
print(out)
print("true labels:", te["label"].head(3).tolist())

# invalid input should raise a clear error
try:
    inference.predict_fn(pd.DataFrame([{"dur": 0.1}]), bundle)
except ValueError as e:
    print("Invalid input handled:", e)