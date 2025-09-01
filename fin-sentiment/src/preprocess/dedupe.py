# src/preprocess/dedupe.py
import jsonlines, hashlib
from datasketch import MinHash, MinHashLSH

def shingles(text, k=5):
    tokens = text.lower().split()
    return {" ".join(tokens[i:i+k]) for i in range(max(1,len(tokens)-k+1))}

def run(infile="data/interim/news_clean.jsonl", outfile="data/interim/news_dedup.jsonl", threshold=0.85):
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    with jsonlines.open(infile) as src, jsonlines.open(outfile, "w") as dst:
        for row in src:
            for sent in row.get("sentences", []):
                mh = MinHash(num_perm=128)
                for sh in shingles(sent):
                    mh.update(sh.encode("utf-8"))
                dupes = lsh.query(mh)
                if dupes:
                    continue
                key = hashlib.md5(sent.encode()).hexdigest()
                lsh.insert(key, mh)
                dst.write({"source": row.get("source"), "url": row.get("url"), "sent": sent, "date": row.get("publish_date")})

if __name__ == "__main__":
    run()
