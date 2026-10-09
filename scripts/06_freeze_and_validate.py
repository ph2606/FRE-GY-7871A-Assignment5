"""Freeze actual dependency closure; reconcile aggregate outputs and local hashes."""
from pathlib import Path
import hashlib
import importlib.metadata as metadata
import json
import sys
from packaging.requirements import Requirement
import nbformat
import pandas as pd
import fitz
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import DATA, OUT

DIRECT = ['numpy','pandas','scipy','scikit-learn','statsmodels','matplotlib','requests',
    'beautifulsoup4','vaderSentiment','nbformat','nbclient','ipykernel','pytest',
    'PyMuPDF','langdetect','transformers','torch','huggingface-hub','safetensors']


def freeze():
    pins = {name: metadata.version(name) for name in DIRECT}
    header = '# CPU PyTorch wheel source; all packages below are exact observed versions.\n--extra-index-url https://download.pytorch.org/whl/cpu\n'
    (ROOT / 'requirements.txt').write_text(header + '\n'.join(f'{name}=={version}' for name, version in pins.items()) + '\n')
    pending, seen = list(DIRECT), {}
    while pending:
        name = pending.pop()
        normalized = name.lower().replace('_','-')
        if normalized in seen:
            continue
        distribution = metadata.distribution(name)
        seen[normalized] = (distribution.metadata['Name'], distribution.version)
        for raw in distribution.requires or []:
            requirement = Requirement(raw)
            if requirement.marker is None or requirement.marker.evaluate({'extra': ''}):
                pending.append(requirement.name)
    lock = header + '\n'.join(f'{name}=={version}' for name, version in sorted(seen.values(), key=lambda x:x[0].lower())) + '\n'
    (ROOT / 'requirements-lock.txt').write_text(lock)


def validate():
    docs = pd.read_csv(DATA / 'scored_texts.csv')
    summary = json.loads((OUT / 'summary.json').read_text())
    assert docs.id.is_unique
    assert docs.source_type.value_counts().to_dict() == summary['documents']
    assert pd.to_datetime(docs.available_at, utc=True).max() <= pd.Timestamp('2026-10-08T20:00:00Z')
    assert not docs.token_truncation.any() and docs.discarded_tokens.eq(0).all()
    assert docs.contextual_chunk_count.ge(1).all()
    nb = nbformat.read(ROOT / 'assignment5.ipynb', as_version=4)
    nbformat.validate(nb)
    cells = [cell for cell in nb.cells if cell.cell_type == 'code']
    assert cells and all(cell.execution_count is not None for cell in cells)
    assert not [o for c in cells for o in c.outputs if o.output_type == 'error']
    pdf = fitz.open(ROOT / 'Panagiotis_Housos_Assignment5.pdf')
    assert len(pdf) == 6, f'Expected six report pages, found {len(pdf)}'
    assert all(len(page.get_text()) > 200 for page in pdf)
    pdf.close()
    local_inputs = sorted((ROOT / 'data/processed').glob('*.csv')) + sorted((ROOT / 'data/processed').glob('*.json'))
    rows = []
    for path in local_inputs:
        rows.append((str(path.relative_to(ROOT)).replace('\\','/'), path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest()))
    text = ['# Submitted snapshot', '', 'As of October 9, 2026; final information cutoff October 8 at 16:00 America/New_York.',
        'Datasets and detailed labels remain local. These hashes identify the submitted snapshot, not an immutable public archive.', '',
        f'Eligible scored documents: {len(docs):,}. Counts: {summary["documents"]}.',
        f'Calendar days: {summary["calendar_days"]}; cash sessions: {summary["market_sessions"]}; assessment sessions: {summary["test_market_sessions"]}.',
        f'Core family: {summary["core_family_size"]} planned tests; {summary["tested_hypotheses"]} have sufficient pairs; {summary["holm_significant"]} Holm-significant at 5%.', '',
        '| Local input | Bytes | SHA256 |', '| --- | ---: | --- |']
    text += [f'| {path} | {size} | {digest} |' for path, size, digest in rows]
    text += ['', 'The executed notebook saves aggregate statistics and figures. The separate six-page PDF is not committed to Git.',
        'The contextual model is pinned to an immutable Hugging Face revision; its own artifact manifest records every downloaded hash.',
        'Sentiment diagnostic agreement is against AI labels, not an independent human gold standard.']
    (ROOT / 'SNAPSHOT.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    print(f'Validated {len(docs)} scored documents, {len(cells)} executed cells and six PDF pages.')


if __name__ == '__main__':
    freeze()
    validate()
