# src/train/split_vectorize.py
import pandas as pd, jsonlines, pathlib
from sklearn.model_selection import train_test_split

def to_df(infile):
    rows=[]
    with jsonlines.open(infile) as f:
        for r in f:
            rows.append({"sent": r["sent"], "label": r["label"], "date": r.get("date")})
    return pd.DataFrame(rows)

def run():
    df = to_df("data/labeled/merged.jsonl").dropna()
    tr, te = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    tr, va = train_test_split(tr, test_size=0.2, stratify=tr["label"], random_state=42)
    pathlib.Path("data/processed").mkdir(parents=True, exist_ok=True)
    tr.to_parquet("data/processed/train.parquet")
    va.to_parquet("data/processed/val.parquet")
    te.to_parquet("data/processed/test.parquet")

if __name__ == "__main__":
    run()
