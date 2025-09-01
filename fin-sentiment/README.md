# fin-sentiment (mini)

A tiny, runnable template to go from raw financial text → labeled sentiment dataset → baseline models.

## Quick start
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# (optional) download spaCy model
python -m spacy download en_core_web_sm

# Run the pipeline on the sample data
python src/preprocess/clean.py
python src/preprocess/dedupe.py
python src/label/weak_supervision.py
python src/label/merge.py
python src/train/split_vectorize.py
python src/train/tfidf_logreg.py
# (optional) FinBERT
# python src/train/finbert_train.py
```

## Structure
```
data/raw            # sample jsonl
data/interim        # cleaned, deduped
data/labeled        # labels (weak/market/gold) and merged
data/processed      # splits parquet
src/                # pipeline code
docs/               # guidelines
```

Created: 2025-09-01
