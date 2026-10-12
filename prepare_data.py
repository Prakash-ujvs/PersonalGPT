"""Download TinyStories dataset and write to corpus.txt.
Usage: python prepare_data.py [n_stories]
"""
import sys
from datasets import load_dataset


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    print(f"Streaming {n:,} stories from TinyStories...")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    with open("corpus.txt", "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            if i >= n:
                break
            f.write(ex["text"].strip() + " ")
            if (i + 1) % 10000 == 0:
                print(f"  {i + 1:,} / {n:,} stories written...")

    print("Successfully wrote corpus.txt")


if __name__ == "__main__":
    main()