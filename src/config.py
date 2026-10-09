from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = '2026-08-01'
TRAIN_END = '2026-08-21'
END = '2026-10-08'
AS_OF = '2026-10-09'
SEED = 787105
# Descriptive names from inspecting this submission's training terms, not returns.
TOPIC_LABELS = {'topic_0': 'War and political commentary', 'topic_1': 'Voting and election campaigning',
                'topic_2': 'Senate primaries and races', 'topic_3': 'Trump, taxes and campaigning'}
DATA = ROOT / 'data/processed'
OUT = ROOT / 'outputs'
for directory in [DATA, OUT]:
    directory.mkdir(parents=True, exist_ok=True)
