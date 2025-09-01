# src/preprocess/clean.py
import re, jsonlines, pathlib, spacy
from ftfy import fix_text

ABBR = {
  r"\bYoY\b": "year over year",
  r"\bQoQ\b": "quarter over quarter",
  r"\bEPS\b": "earnings per share",
  r"\bEBITDA\b": "earnings before interest taxes depreciation and amortization",
}

def normalize(text):
    t = fix_text(text)
    t = re.sub(r"\s+", " ", t).strip()
    for k,v in ABBR.items():
        t = re.sub(k, v, t, flags=re.I)
    return t

def run(infile="data/raw/news.jsonl", outfile="data/interim/news_clean.jsonl"):
    nlp = spacy.load("en_core_web_sm", disable=["ner","tagger","lemmatizer"])
    pathlib.Path("data/interim").mkdir(parents=True, exist_ok=True)
    with jsonlines.open(infile) as src, jsonlines.open(outfile, "w") as dst:
        for row in src:
            text = ((row.get("title","") + ". " + row.get("text",""))).strip()
            text = normalize(text)
            doc = nlp(text)
            sents = [s.text.strip() for s in doc.sents if 10 <= len(s.text) <= 280]
            if sents:
                row["sentences"] = sents
                dst.write(row)

if __name__ == "__main__":
    run()
