import pandas as pd

tr = pd.read_csv("../data/processed/train.csv")
te = pd.read_csv("../data/processed/test.csv")

print("train:", tr.shape, "test:", te.shape)
print("Same columns:", list(tr.columns) == list(te.columns))
print(tr["label"].value_counts(normalize=True))
print("Missing values (train, test):", tr.isna().sum().sum(), te.isna().sum().sum())

assert list(tr.columns) == list(te.columns), "Train/test columns differ"
assert tr.isna().sum().sum() == 0 and te.isna().sum().sum() == 0, "Missing values found"
assert te.shape[0] == 82332, "Test set row count changed"
assert tr["label"].isin([0, 1]).all(), "Unexpected label values"
print("All checks passed")