"""pip install datasets   then:  python prepare_data.py 50000   (stories -> corpus.txt, ~35MB)"""
import sys
from datasets import load_dataset
n = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
with open("corpus.txt", "w", encoding="utf-8") as f:
    for i, ex in enumerate(ds):
        if i >= n: break
        f.write(ex["text"].strip() + " ")
print("wrote corpus.txt")