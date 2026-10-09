"""Pinned CardiffNLP contextual wording valence; no political-stance inference.

Model weights and provenance stay in data/raw. VADER columns are never modified.
Long documents retain every token through nonoverlapping chunks, with payload
token-count-weighted averaging of the three predicted class probabilities.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata
import json
import re
import time

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
REVISION = "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7"
MODEL_DIR = ROOT / "data/raw/models/cardiffnlp_twitter_roberta_sentiment" / REVISION
MODEL_FILES = ["README.md", "config.json", "merges.txt", "special_tokens_map.json",
               "vocab.json", "pytorch_model.bin"]
CLASS_NAMES = ["negative", "neutral", "positive"]
MAX_SEQUENCE_TOKENS = 512
PAYLOAD_TOKENS = 510


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def acquire_model() -> dict:
    """Download official immutable-revision assets, hash-check on every reuse."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    api_path = MODEL_DIR / "huggingface_model_info.json"
    api_url = f"https://huggingface.co/api/models/{MODEL_ID}/revision/{REVISION}?blobs=true"
    if not api_path.exists():
        response = requests.get(api_url, timeout=60)
        response.raise_for_status()
        api_path.write_bytes(response.content)
        write_json(MODEL_DIR / "huggingface_model_info.metadata.json", {
            "url": api_url, "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "sha256": sha256_file(api_path), "http_status": response.status_code})
    info = json.loads(api_path.read_text(encoding="utf-8"))
    api_metadata = json.loads((MODEL_DIR / "huggingface_model_info.metadata.json").read_text(encoding="utf-8"))
    if sha256_file(api_path) != api_metadata["sha256"]:
        raise ValueError("Cached model-info provenance hash mismatch.")
    if info["sha"] != REVISION:
        raise ValueError("Model metadata revision does not match immutable pin.")
    expected = {s["rfilename"]: s.get("lfs", {}).get("sha256") for s in info["siblings"]}

    def download(filename):
        path = MODEL_DIR / filename
        metadata_path = MODEL_DIR / (filename + ".metadata.json")
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{filename}"
        if not path.exists():
            temporary = MODEL_DIR / (filename + ".partial")
            for attempt in range(3):
                try:
                    with requests.get(url, stream=True, timeout=(30, 90)) as response:
                        response.raise_for_status()
                        with temporary.open("wb") as handle:
                            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                                if chunk:
                                    handle.write(chunk)
                        actual = sha256_file(temporary)
                        if expected.get(filename) and actual != expected[filename]:
                            raise ValueError(f"Official LFS hash mismatch: {filename}")
                        temporary.replace(path)
                        write_json(metadata_path, {"url": url, "revision": REVISION,
                            "retrieved_at": datetime.now(timezone.utc).isoformat(),
                            "sha256": actual, "bytes": path.stat().st_size,
                            "http_status": response.status_code,
                            "source_date_header": response.headers.get("Date"),
                            "official_lfs_sha256": expected.get(filename)})
                    print(f"Acquired {filename}: {path.stat().st_size:,} bytes", flush=True)
                    break
                except (requests.RequestException, ValueError):
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if sha256_file(path) != metadata["sha256"]:
            raise ValueError(f"Model-file cache hash mismatch: {filename}")
        if expected.get(filename) and metadata["sha256"] != expected[filename]:
            raise ValueError(f"Model cache differs from official LFS hash: {filename}")
        return metadata

    with ThreadPoolExecutor(max_workers=4) as pool:
        assets = list(pool.map(download, MODEL_FILES))
    manifest = {"model_id": MODEL_ID, "revision": REVISION,
        "license": info.get("cardData", {}).get("license", "cc-by-4.0"),
        "model_card": f"https://huggingface.co/{MODEL_ID}/blob/{REVISION}/README.md",
        "paper": "https://aclanthology.org/2022.acl-demo.25/",
        "model_info_url": api_url, "model_info_sha256": sha256_file(api_path),
        "assets": assets, "remote_code": False,
        "weight_loading": "torch.load(weights_only=True); trusted model architecture from installed Transformers"}
    write_json(MODEL_DIR / "model_manifest.json", manifest)
    return manifest


def preprocess(text: str) -> str:
    """Official-card normalization of social user mentions and HTTP links."""
    value = re.sub(r"https?://\S+", "http", str(text), flags=re.IGNORECASE)
    return re.sub(r"(?<!\w)@\w+", "@user", value)


def token_chunks(token_ids: list[int], payload_tokens: int = PAYLOAD_TOKENS) -> list[list[int]]:
    if payload_tokens < 1:
        raise ValueError("Chunk size must be positive.")
    return [token_ids[i:i + payload_tokens] for i in range(0, len(token_ids), payload_tokens)] or [[]]


def evaluate_review(scores: pd.DataFrame, documents: pd.DataFrame) -> dict | None:
    """Compare to fixed AI development labels; never fit or calibrate to them."""
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    labels_path = ROOT / "data/processed/sentiment_review_labels.csv"
    if not labels_path.exists():
        return None
    labels = pd.read_csv(labels_path)
    missing_ids = set(labels.id) - set(scores.id)
    joined = labels.merge(scores, on="id", how="inner", validate="one_to_one")
    baseline_field = "vader_class" if "vader_class" in documents else "sentiment_class"
    baseline = documents[["id", baseline_field]].rename(columns={baseline_field: "vader_class"})
    joined = joined.merge(baseline, on="id", how="left", validate="one_to_one")
    joined["contextual_agreement"] = joined.ai_reference_class.eq(joined.contextual_class)
    joined["vader_agreement"] = joined.ai_reference_class.eq(joined.vader_class)
    joined.to_csv(ROOT / "data/processed/contextual_review_comparison.csv", index=False)

    def metrics(group, prediction="contextual_class"):
        return {"n": len(group), "agreements": int(group.ai_reference_class.eq(group[prediction]).sum()),
            "ai_reference_agreement": float(accuracy_score(group.ai_reference_class, group[prediction])),
            "ai_reference_macro_f1": float(f1_score(group.ai_reference_class, group[prediction],
                labels=CLASS_NAMES, average="macro", zero_division=0))}

    summary = {**metrics(joined), "contextual": metrics(joined),
        "vader": metrics(joined, "vader_class"),
        "frozen_review_n": len(labels), "eligible_review_n": len(joined),
        "excluded_review_ids": sorted(missing_ids),
        "vader_baseline_field": baseline_field,
        "by_source": {str(k): metrics(g) for k, g in joined.groupby("source_type")},
        "by_split": {str(k): metrics(g) for k, g in joined.groupby("review_split")},
        "confusion_order": CLASS_NAMES,
        "contextual_confusion": confusion_matrix(joined.ai_reference_class, joined.contextual_class, labels=CLASS_NAMES).tolist(),
        "vader_confusion": confusion_matrix(joined.ai_reference_class, joined.vader_class, labels=CLASS_NAMES).tolist(),
        "labels_sha256": sha256_file(labels_path),
        "model_revision": REVISION, "model_fitted_on_review": False,
        "interpretation": "Agreement with the final-corpus-eligible subset of 60 frozen AI-reference development labels only; excluded labels remain frozen and are not replaced. Not independent human-validated accuracy, representative corpus accuracy, political stance validation, or calibrated probability. This diagnostic was consulted for model comparison and cannot serve as an untouched final test set."}
    ambiguous = joined.ambiguous_valence.astype(str).str.lower().eq("true")
    if (~ambiguous).any():
        summary["unambiguous"] = metrics(joined.loc[~ambiguous])
    write_json(ROOT / "outputs/contextual_review_summary.json", summary)
    return summary


def score_documents(batch_size: int = 24, threads: int = 6) -> pd.DataFrame:
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    input_path = ROOT / "data/processed/scored_texts.csv"
    documents = pd.read_csv(input_path, low_memory=False)
    if documents.id.duplicated().any() or documents.analysis_text.isna().any():
        raise ValueError("Unique ids and nonmissing analysis_text are required.")
    input_sha = sha256_file(input_path)
    model_manifest = acquire_model()
    output_path = ROOT / "data/processed/contextual_scores.csv"
    if output_path.exists():
        prior_method_path = ROOT / "outputs/contextual_method.json"
        if prior_method_path.exists():
            prior_method = json.loads(prior_method_path.read_text(encoding="utf-8"))
            if sha256_file(output_path) != prior_method["output_sha256"]:
                raise ValueError("Contextual score cache does not match its provenance hash.")
        cached = pd.read_csv(output_path)
        current_hashes = documents.set_index("id").analysis_text.map(
            lambda t: hashlib.sha256(str(t).encode("utf-8")).hexdigest())
        if (not cached.id.duplicated().any() and set(documents.id).issubset(set(cached.id))
                and cached.model_revision.eq(REVISION).all()
                and cached.set_index("id").analysis_text_sha256.reindex(current_hashes.index).eq(current_hashes).all()):
            cache_source_n = len(cached)
            cached = cached.set_index("id").loc[documents.id].reset_index()
            return finalize_scores(cached, documents, model_manifest, input_sha,
                batch_size, threads, cache_reused=True, cache_source_n=cache_source_n)
    torch.set_num_threads(threads)
    torch.manual_seed(787105)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True,
                                             model_max_length=MAX_SEQUENCE_TOKENS)
    config = AutoConfig.from_pretrained(str(MODEL_DIR), local_files_only=True)
    labels = [str(config.id2label[i]).lower() for i in range(3)]
    if labels != CLASS_NAMES:
        raise ValueError(f"Unexpected model label order: {labels}")
    model = AutoModelForSequenceClassification.from_config(config)
    # The pinned official repository provides a PyTorch state dictionary. Load
    # tensors only, without arbitrary pickle object construction or remote code.
    state = torch.load(MODEL_DIR / "pytorch_model.bin", map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    allowed_buffers = {"roberta.embeddings.position_ids", "roberta.embeddings.token_type_ids"}
    # The official checkpoint includes the base encoder's unused pooler. The
    # sentiment classification head uses token states and never calls that pooler.
    allowed_unused = allowed_buffers | {"roberta.pooler.dense.weight", "roberta.pooler.dense.bias"}
    if set(incompatible.missing_keys) - allowed_buffers or set(incompatible.unexpected_keys) - allowed_unused:
        raise ValueError(f"Unexpected weight incompatibility: {incompatible}")
    del state
    model.eval()
    records = []
    chunks = []
    for position, row in enumerate(documents.itertuples(index=False)):
        text = str(row.analysis_text)
        payload = tokenizer.encode(preprocess(text), add_special_tokens=False, truncation=False)
        pieces = token_chunks(payload)
        records.append({"id": row.id, "analysis_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "contextual_tokens": len(payload), "contextual_chunk_count": len(pieces),
            "contextual_chunked": len(pieces) > 1, "token_truncation": False,
            "discarded_tokens": 0, "model_revision": REVISION})
        for piece in pieces:
            chunks.append((position, max(len(piece), 1), tokenizer.build_inputs_with_special_tokens(piece)))
    # Similar-length batches avoid padding 12-word headlines to long comments.
    chunks.sort(key=lambda x: len(x[2]))
    totals = np.zeros((len(documents), 3), dtype=np.float64)
    token_totals = np.zeros(len(documents), dtype=np.float64)
    with torch.inference_mode():
        for offset in range(0, len(chunks), batch_size):
            batch = chunks[offset:offset + batch_size]
            encoded = tokenizer.pad({"input_ids": [x[2] for x in batch]}, padding=True,
                                    return_attention_mask=True, return_tensors="pt")
            if encoded["input_ids"].shape[1] > MAX_SEQUENCE_TOKENS:
                raise ValueError("Chunk exceeds model context limit.")
            probabilities = torch.softmax(model(**encoded).logits, dim=-1).cpu().numpy()
            for (position, weight, _), probability in zip(batch, probabilities):
                totals[position] += weight * probability
                token_totals[position] += weight
            if offset % (batch_size * 20) == 0 or offset + batch_size >= len(chunks):
                print(f"Contextual scoring {min(offset + batch_size, len(chunks))}/{len(chunks)} chunks", flush=True)
    probabilities = totals / token_totals[:, None]
    output = pd.DataFrame(records)
    for index, label in enumerate(CLASS_NAMES):
        output[f"contextual_{label}"] = probabilities[:, index]
    output["contextual_valence"] = output.contextual_positive - output.contextual_negative
    output["contextual_class"] = np.asarray(CLASS_NAMES)[probabilities.argmax(axis=1)]
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6):
        raise ValueError("Class probabilities fail normalization.")
    return finalize_scores(output, documents, model_manifest, input_sha,
        batch_size, threads, unused_checkpoint_keys=list(incompatible.unexpected_keys))


def finalize_scores(output: pd.DataFrame, documents: pd.DataFrame, model_manifest: dict,
                    input_sha: str, batch_size: int, threads: int,
                    cache_reused: bool = False, unused_checkpoint_keys: list | None = None,
                    cache_source_n: int | None = None) -> pd.DataFrame:
    """Save unchanged neural outputs and repeatable fixed-label diagnostics."""
    input_path = ROOT / "data/processed/scored_texts.csv"
    output_path = ROOT / "data/processed/contextual_scores.csv"
    if output_path.exists():
        old = pd.read_csv(output_path)
        if len(old) != len(output) or set(old.id) != set(output.id):
            history = ROOT / "data/raw/contextual_score_versions"
            history.mkdir(parents=True, exist_ok=True)
            old_sha = sha256_file(output_path)
            (history / f"contextual_scores_{old_sha[:12]}.csv").write_bytes(output_path.read_bytes())
            prior_method = ROOT / "outputs/contextual_method.json"
            if prior_method.exists():
                (history / f"contextual_scores_{old_sha[:12]}.metadata.json").write_bytes(prior_method.read_bytes())
    output.to_csv(output_path, index=False)
    summary = evaluate_review(output, documents)
    write_json(ROOT / "outputs/contextual_method.json", {
        "model": model_manifest, "input": str(input_path.relative_to(ROOT)).replace("\\", "/"),
        "input_sha256": input_sha, "output_sha256": sha256_file(output_path),
        "n": len(output), "chunks": int(output.contextual_chunk_count.sum()), "chunked_documents": int(output.contextual_chunked.sum()),
        "discarded_tokens": 0, "payload_tokens_per_chunk": PAYLOAD_TOKENS,
        "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
        "aggregation": "Nonoverlapping full-document payload-token chunks; three-class probabilities averaged with each chunk's payload token count. No tokens discarded. Cross-chunk context is unavailable.",
        "preprocessing": "HTTP links->http and social @mentions->@user, following official model-card conventions",
        "score": "p(positive)-p(negative), range[-1,1]; class=argmax of aggregated class probabilities",
        "calibration": "Uncalibrated model outputs in the 2026 political/news domain; wording valence is not partisan support, voter intention, or probability of election outcome",
        "training": "Off-the-shelf model, no fine-tuning, threshold tuning, market outcomes, or review labels used during inference",
        "versions": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "tokenizers", "huggingface-hub", "numpy"]},
        "threads": threads, "batch_size": batch_size,
        "cache_reused": cache_reused,
        "cache_source_n": cache_source_n,
        "unused_checkpoint_keys": unused_checkpoint_keys or ["roberta.pooler.dense.weight", "roberta.pooler.dense.bias", "roberta.embeddings.position_ids"],
        "retrieved_and_scored_at": datetime.now(timezone.utc).isoformat(),
        "review_summary": summary,
    })
    print(json.dumps({"documents": len(output), "chunks": int(output.contextual_chunk_count.sum()),
        "chunked_documents": int(output.contextual_chunked.sum()),
        "review_agreement": summary["ai_reference_agreement"] if summary else None}, indent=2))
    return output
