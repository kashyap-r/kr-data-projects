# src/label/weak_supervision.py
import jsonlines

POS = {"beat","beats","surge","rise","rises","upgrade","upgraded","record","strong"}
NEG = {"miss","misses","plunge","fall","falls","lawsuit","probe","downgrade","downgraded","warning"}

def label(sent):
    s = sent.lower()
    pos = any(w in s for w in POS)
    neg = any(w in s for w in NEG)
    if pos and not neg: return "positive", 0.7
    if neg and not pos: return "negative", 0.7
    return "neutral", 0.55

def run(infile="data/interim/news_dedup.jsonl", outfile="data/labeled/weak_labels.jsonl"):
    with jsonlines.open(infile) as src, jsonlines.open(outfile, "w") as dst:
        for row in src:
            y, conf = label(row["sent"])
            dst.write({**row, "label": y, "conf": conf, "strategy": "weak"})

if __name__ == "__main__":
    run()
