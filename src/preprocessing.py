import os
import pandas as pd

# RAW_A = "data/raw/UNSW_NB15_training-set.csv"
# RAW_B = "data/raw/UNSW_NB15_testing-set.csv"
# OUT_DIR = "data/processed"

RAW_A = "../data/raw/UNSW_NB15_training-set.csv"
RAW_B = "../data/raw/UNSW_NB15_testing-set.csv"
OUT_DIR= "../data/processed"


def load_split(path_a, path_b):
    """Return (train, test), assigning the larger file to train."""
    a, b = pd.read_csv(path_a), pd.read_csv(path_b)
    return (a, b) if len(a) > len(b) else (b, a)


def preprocess(train, test):
    # drop the index column and the multi-class target (leakage for binary label)
    train = train.drop(columns=["id", "attack_cat"]).drop_duplicates().copy()
    test = test.drop(columns=["id", "attack_cat"]).copy()

    cat_cols = ["proto", "service", "state"]

    # proto has ~130 values: keep the 10 most common, group the rest
    top_proto = train["proto"].value_counts().nlargest(10).index
    for df in (train, test):
        df["proto"] = df["proto"].where(df["proto"].isin(top_proto), "other")

    train = pd.get_dummies(train, columns=cat_cols, dtype=int)
    test = pd.get_dummies(test, columns=cat_cols, dtype=int)

    # make test columns identical to train (unseen categories become 0)
    test = test.reindex(columns=train.columns, fill_value=0)
    return train, test


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    train, test = load_split(RAW_A, RAW_B)
    print("Raw train:", train.shape, "Raw test:", test.shape)

    train_p, test_p = preprocess(train, test)

    # label first (the layout SageMaker's built-in XGBoost expects)
    cols = ["label"] + [c for c in train_p.columns if c != "label"]
    train_p[cols].to_csv(f"{OUT_DIR}/train.csv", index=False)
    test_p[cols].to_csv(f"{OUT_DIR}/test.csv", index=False)

    print("Processed train:", train_p.shape, "Processed test:", test_p.shape)
    print(train_p["label"].value_counts())