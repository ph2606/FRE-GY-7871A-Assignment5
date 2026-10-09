"""Acquire public prediction-market histories and adjusted equity closes."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.market_data import collect

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2026-08-01")
    parser.add_argument("--end", default="2026-10-08")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    collect(args.start, args.end, args.refresh)
