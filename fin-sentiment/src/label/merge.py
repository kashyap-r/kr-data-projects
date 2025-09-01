# src/label/merge.py
import jsonlines, collections

def load(path):
    with jsonlines.open(path) as f:
        for r in f:
            yield r

def run(paths=("data/labeled/weak_labels.jsonl",), out="data/labeled/merged.jsonl"):
    # precedence order already handled by order in 'paths'
    merged = collections.OrderedDict()
    for p in paths:
        for r in load(p):
            key = (r.get("url"), r.get("sent"))
            if key not in merged:
                merged[key] = r
    with jsonlines.open(out, "w") as dst:
        for r in merged.values():
            dst.write(r)

if __name__ == "__main__":
    run()
