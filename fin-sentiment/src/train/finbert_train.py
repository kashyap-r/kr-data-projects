# src/train/finbert_train.py
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

MODEL = "ProsusAI/finbert"
LABEL2ID = {"negative":0, "neutral":1, "positive":2}

def load_split():
    tr = pd.read_parquet("data/processed/train.parquet")
    va = pd.read_parquet("data/processed/val.parquet")
    te = pd.read_parquet("data/processed/test.parquet")
    for df in (tr,va,te): df["labels"] = df["label"].map(LABEL2ID)
    tok = AutoTokenizer.from_pretrained(MODEL)
    def tok_fn(batch): return tok(batch["sent"], truncation=True, padding=True, max_length=128)
    ds = DatasetDict({
        "train": Dataset.from_pandas(tr[["sent","labels"]]),
        "validation": Dataset.from_pandas(va[["sent","labels"]]),
        "test": Dataset.from_pandas(te[["sent","labels"]]),
    }).map(tok_fn, batched=True)
    return ds

def metrics(p):
    import numpy as np
    preds = p.predictions.argmax(-1); labels = p.label_ids
    return {"acc": (preds==labels).mean().item(), "f1_macro": f1_score(labels, preds, average="macro")}

if __name__ == "__main__":
    ds = load_split()
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=3)
    args = TrainingArguments(
        output_dir="runs/finbert",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=1,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=10
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["validation"])
    trainer.train()
    print(trainer.evaluate(ds["test"]))
