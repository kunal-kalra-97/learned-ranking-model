"""Merge two feature JSON files into one training set."""
import json
import sys

f1 = sys.argv[1] if len(sys.argv) > 1 else "datasets/train_features_1.json"
f2 = sys.argv[2] if len(sys.argv) > 2 else "datasets/train_features_2.json"
out = sys.argv[3] if len(sys.argv) > 3 else "datasets/train_features.json"

with open(f1) as fh: d1 = json.load(fh)
with open(f2) as fh: d2 = json.load(fh)
merged = d1 + d2
with open(out, "w") as fh: json.dump(merged, fh)
print(f"Merged {len(d1)} + {len(d2)} = {len(merged)} examples -> {out}")