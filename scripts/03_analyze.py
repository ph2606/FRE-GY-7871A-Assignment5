from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.text_model import score_documents
from src.analysis import analyze
from src.issue_sentiment import build as issue_indices

if __name__ == '__main__':
    score_documents()
    _, _, results = analyze()
    issue_indices()
    print(results.loc[results['index'].eq('index_all')].to_string(index=False))
