"""Run from the repository root; all retrieval bytes are locally cached."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.collect_text import main

if __name__ == '__main__':
    main()
