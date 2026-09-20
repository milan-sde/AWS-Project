import os
import shutil
import pandas as pd

A = "data/raw/UNSW_NB15_training-set.csv"
B = "data/raw/UNSW_NB15_testing-set.csv"
OUT = "data/s3_upload"

n_a, n_b = len(pd.read_csv(A)), len(pd.read_csv(B))
train_raw, test_raw = (A, B) if n_a > n_b else (B, A)
print("raw train rows:", max(n_a, n_b), "| raw test rows:", min(n_a, n_b))

files = {
    f"{OUT}/raw/unsw_nb15/train/train.csv": train_raw,
    f"{OUT}/raw/unsw_nb15/test/test.csv": test_raw,
    f"{OUT}/processed/train/train.csv": "data/processed/train.csv",
    f"{OUT}/processed/test/test.csv": "data/processed/test.csv",
}
for dest, src in files.items():
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copy(src, dest)
    print("staged:", dest)