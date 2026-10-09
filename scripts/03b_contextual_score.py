"""Acquire a pinned contextual sentiment model and score without changing VADER."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.contextual_sentiment import acquire_model, score_documents

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquire-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    if args.acquire_only:
        result = acquire_model()
        print(f"Pinned model ready: {result['revision']}")
    else:
        score_documents(args.batch_size, args.threads)
