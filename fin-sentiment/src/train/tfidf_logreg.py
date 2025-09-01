# src/train/tfidf_logreg.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

def load(p): return pd.read_parquet(p)

if __name__ == "__main__":
    tr = load("data/processed/train.parquet")
    va = load("data/processed/val.parquet")
    te = load("data/processed/test.parquet")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.95)),
        ("clf", LogisticRegression(max_iter=300, class_weight="balanced"))
    ])
    pipe.fit(tr.sent, tr.label)

    for name, df in [("Val", va), ("Test", te)]:
        preds = pipe.predict(df.sent)
        print(f"\n{name}:\n" + classification_report(df.label, preds, digits=3))
