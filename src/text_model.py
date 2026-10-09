"""Frozen temporal NMF topic model and transparent lexical valence indices.

Topic probabilities below are normalized NMF loadings, not calibrated posterior
probabilities. Valence measures wording, not support for either party.
"""
import json
import hashlib
import re
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from .config import DATA, OUT, START, TRAIN_END, END, SEED

STOP = sorted(set(ENGLISH_STOP_WORDS) | set('midterm midterms election elections said says say new news read www com https http amp just like don people year years 2026 2027'.split()))


def normalize_loadings(loadings):
    values = np.asarray(loadings, dtype=float)
    totals = values.sum(axis=1, keepdims=True)
    return np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)


def align_to_closes(available_at, closes):
    """First actual close at or after availability; never compress missing days."""
    available = pd.to_datetime(available_at, utc=True, format='mixed', errors='coerce')
    close_index = pd.DatetimeIndex(pd.to_datetime(closes, utc=True))
    if not close_index.is_monotonic_increasing or not close_index.is_unique:
        raise ValueError('Closes must be sorted and unique.')
    positions = close_index.searchsorted(available)
    result = pd.Series(pd.NaT, index=getattr(available_at, 'index', None), dtype='datetime64[ns, UTC]')
    valid = available.notna() & (positions < len(close_index))
    result.loc[valid] = close_index[positions[valid]]
    return result


def npmi_score(matrix, components, top_n=8, sample_weight=None):
    binary = (matrix > 0).astype(float)
    n = matrix.shape[0]
    weight = np.ones(n) / n if sample_weight is None else np.asarray(sample_weight) / np.sum(sample_weight)
    prevalence = np.asarray(binary.T @ weight).ravel()
    scores = []
    for component in components:
        terms = component.argsort()[-top_n:]
        pairs = []
        for i, a in enumerate(terms):
            for b in terms[i + 1:]:
                joint = (binary[:, a].multiply(binary[:, b]).T @ weight).item()
                if joint == 0:
                    pairs.append(-1.)
                elif joint == 1:
                    pairs.append(0.)
                else:
                    pairs.append(np.log(joint / (prevalence[a] * prevalence[b])) / -np.log(joint))
        scores.append(float(np.mean(pairs)))
    return float(np.mean(scores))


def fit_model(training):
    # Equal objective weight per medium retains the scarce social training text.
    balanced = pd.concat([g.sample(n=min(1500, len(g)), random_state=SEED) for _, g in training.groupby('source_type')])
    if len(balanced) < 50:
        raise ValueError('Too few training documents for a defensible topic model.')
    vectorizer = TfidfVectorizer(stop_words=STOP, lowercase=True, strip_accents='unicode',
        min_df=5, max_df=.92, max_features=8000, ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform(balanced.analysis_text)
    counts = balanced.source_type.map(balanced.source_type.value_counts()).to_numpy()
    row_weight = len(balanced) / (balanced.source_type.nunique() * counts)
    weighted_matrix = matrix.multiply(np.sqrt(row_weight)[:, None]).tocsr()
    candidates = []
    models = {}
    for k in [4, 6, 8]:
        model = NMF(n_components=k, init='nndsvda', random_state=SEED, max_iter=800)
        model.fit(weighted_matrix)
        loadings = model.transform(matrix)
        alternate = NMF(n_components=k, init='random', random_state=SEED + 1, max_iter=800).fit(weighted_matrix)
        similarity = cosine_similarity(model.components_, alternate.components_)
        rows, cols = linear_sum_assignment(-similarity)
        normalized = normalize_loadings(loadings)
        shares = np.average(normalized, axis=0, weights=row_weight)
        contributions = normalized * row_weight[:, None]
        mass = contributions.sum(axis=0)
        ess = mass ** 2 / (contributions ** 2).sum(axis=0)
        top_three = np.sort(contributions, axis=0)[-3:].sum(axis=0) / mass
        candidates.append({'k': k, 'training_npmi': npmi_score(matrix, model.components_, sample_weight=row_weight),
            'seed_stability_cosine': float(similarity[rows, cols].mean()),
            'smallest_mean_weight': float(shares.min()), 'min_topic_effective_support': float(ess.min()),
            'max_topic_top_three_mass': float(top_three.max()), 'training_documents': len(balanced)})
        models[k] = model
    table = pd.DataFrame(candidates)
    # Fixed selection rule uses training text only, with no market variables.
    table['eligible'] = (table.smallest_mean_weight >= .02) & (table.min_topic_effective_support >= 20) & (table.max_topic_top_three_mass <= .35)
    eligible = table.loc[table.eligible]
    if eligible.empty:
        raise ValueError('No candidate passes training-text support diagnostics; inspect the corpus.')
    selected = int(eligible.sort_values(['training_npmi', 'k'], ascending=[False, True]).iloc[0].k)
    table['selected'] = table.k.eq(selected)
    table.to_csv(OUT / 'topic_model_selection.csv', index=False)
    return vectorizer, models[selected], table


def score_documents(use_contextual=True):
    documents = pd.read_csv(DATA / 'texts.csv', low_memory=False).fillna('')
    required = {'id', 'source', 'source_type', 'published_at', 'available_at', 'text', 'title'}
    if not required.issubset(documents):
        raise ValueError(f'Missing text fields: {required - set(documents)}')
    documents['published_at'] = pd.to_datetime(documents.published_at, utc=True, format='mixed', errors='coerce')
    documents['available_at'] = pd.to_datetime(documents.available_at, utc=True, format='mixed', errors='coerce')
    documents['analysis_text'] = np.where(documents.source_type.eq('news'), documents.title, documents.text)
    documents['canonical'] = documents.analysis_text.str.lower().str.replace(r'[^a-z0-9]+', ' ', regex=True).str.strip()
    documents = documents.sort_values(['available_at', 'id'], kind='stable')
    documents['exclusion'] = ''
    def exclude(mask, reason):
        documents.loc[mask & documents.exclusion.eq(''), 'exclusion'] = reason
    dates = documents.published_at.dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d')
    exclude(documents.published_at.isna() | documents.available_at.isna(), 'missing_clock')
    exclude(dates.lt(START) | dates.gt(END), 'outside_window')
    exclude(documents.available_at.lt(documents.published_at), 'invalid_availability')
    final_close = pd.Timestamp(END, tz='America/New_York') + pd.Timedelta(hours=16)
    exclude(documents.available_at.gt(final_close), 'after_final_information_cutoff')
    exclude(documents.analysis_text.str.findall(r'[A-Za-z]+').str.len().lt(5), 'under_five_words')
    exclude(documents.duplicated(['source_type', 'canonical']), 'duplicate_text_within_medium')
    documents.groupby(['source_type', 'exclusion']).size().rename('n').reset_index().to_csv(OUT / 'text_filter_audit.csv', index=False)
    documents = documents.loc[documents.exclusion.eq('')].copy().reset_index(drop=True)
    train_cutoff = pd.Timestamp(TRAIN_END, tz='America/New_York') + pd.Timedelta(hours=16)
    training = documents.loc[documents.available_at <= train_cutoff]
    vectorizer, model, candidates = fit_model(training)
    weights = normalize_loadings(model.transform(vectorizer.transform(documents.analysis_text)))
    terms = vectorizer.get_feature_names_out()
    topic_rows = []
    for j, component in enumerate(model.components_):
        top = terms[component.argsort()[-12:][::-1]]
        topic_rows.append({'topic': f'topic_{j}', 'label': ' / '.join(top[:3]), 'top_terms': ', '.join(top)})
        documents[f'topic_{j}'] = weights[:, j]
    pd.DataFrame(topic_rows).to_csv(OUT / 'topic_terms.csv', index=False)
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(t) for t in documents.analysis_text]
    documents['sentiment'] = [v['compound'] for v in scores]
    documents['polarity_balance'] = [v['pos'] - v['neg'] for v in scores]
    documents['sentiment_class'] = np.select([documents.sentiment.le(-.05), documents.sentiment.ge(.05)], ['negative', 'positive'], default='neutral')
    documents['vader'] = documents.sentiment
    documents['vader_class'] = documents.sentiment_class
    sentiment_method = 'VADER 3.3.2 compound'
    contextual_path = DATA / 'contextual_scores.csv'
    if use_contextual and contextual_path.exists():
        contextual = pd.read_csv(contextual_path)
        if contextual.id.duplicated().any() or set(contextual.id) != set(documents.id):
            raise ValueError('Contextual scores do not match current corpus IDs; rerun contextual scoring.')
        current_hashes = documents.set_index('id').analysis_text.map(lambda text: hashlib.sha256(text.encode('utf-8')).hexdigest())
        if not contextual.set_index('id').analysis_text_sha256.reindex(current_hashes.index).eq(current_hashes).all():
            raise ValueError('Contextual scores refer to changed text; rerun contextual scoring.')
        documents = documents.merge(contextual, on='id', validate='one_to_one')
        documents['sentiment'] = documents.contextual_valence
        documents['sentiment_class'] = documents.contextual_class
        sentiment_method = 'Pinned CardiffNLP Twitter RoBERTa, P(positive)-P(negative); token-weighted full-text chunks'
    documents['model_covered'] = weights.sum(axis=1) > 0
    documents['training'] = documents.available_at <= train_cutoff
    documents.to_csv(DATA / 'scored_texts.csv', index=False)
    specification = {'training_end': str(train_cutoff), 'training_available_n': len(training),
        'fitted_balanced_n': int(candidates.training_documents.iloc[0]), 'k': model.n_components,
        'source_balance': 'All capped training documents retained; sqrt(N/(medium_count*n_medium)) row scaling balances the NMF objective. Vocabulary IDF uses all training documents.',
        'stop_words': STOP, 'selection': 'Highest training top-eight-term NPMI among K=4,6,8 with min mean loading >=.02, every topic effective support >=20 and top-three-document mass <=.35; tie smaller K.',
        'development_diagnostic': 'Initial min_df=3 fits isolated repeated quotations from one HN thread. Training-text-only review raised min_df to5 and added concentration screens. This is retrospective method development, not preregistration.',
        'weights': 'Nonnegative NMF document loadings normalized to sum one; zero rows unclassified.',
        'sentiment': sentiment_method,
        'vader_robustness': 'VADER compound, negative <=-.05; positive >=.05. Polarity balance uses pos-neg.',
        'accuracy': 'No independent human gold standard; API topic relevance and lexical target errors remain.',
        'timing': 'First 16:00 America/New_York close at or after available_at; market analysis uses actual trading closes.'}
    (OUT / 'text_method.json').write_text(json.dumps(specification, indent=2))
    return documents


def aggregate_indices(documents, closes, score='sentiment', equal_outlets=False):
    topics = [c for c in documents if re.fullmatch(r'topic_\d+', c)]
    frame = documents.loc[documents.model_covered].copy()
    frame['close'] = align_to_closes(frame.available_at, closes)
    frame = frame.dropna(subset=['close'])
    records = []
    for (close, kind), group in frame.groupby(['close', 'source_type']):
        record = {'close': close, 'source_type': kind, 'n': len(group)}
        if equal_outlets:
            entities = group.get('author', pd.Series('', index=group.index)) if kind == 'social' else group.source
            entities = entities.replace('', np.nan).fillna(group.id)
            base_weights = 1 / entities.map(entities.value_counts())
        else:
            base_weights = pd.Series(1., index=group.index)
        record['all'] = float(np.average(group[score], weights=base_weights))
        for topic in topics:
            weight = group[topic] * base_weights
            record[topic] = float(np.average(group[score], weights=weight)) if weight.sum() >= .5 else np.nan
            record[topic + '_mass'] = float(group[topic].sum())
        records.append(record)
    table = pd.DataFrame(records)
    index = pd.DatetimeIndex(pd.to_datetime(closes, utc=True))
    result = pd.DataFrame(index=index)
    result.index.name = 'close'
    for kind in ['news', 'social']:
        subset = table.loc[table.source_type.eq(kind)].set_index('close') if len(table) else pd.DataFrame()
        for topic in ['all'] + topics:
            result[f'{kind}_{topic}'] = subset[topic].reindex(index) if topic in subset else np.nan
        result[f'{kind}_n'] = subset['n'].reindex(index) if 'n' in subset else 0
    for topic in ['all'] + topics:
        result[f'index_{topic}'] = (result[f'news_{topic}'] + result[f'social_{topic}']) / 2
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lexical-only', action='store_true', help='Establish current eligible texts before contextual scoring or refreshing.')
    args = parser.parse_args()
    data = score_documents(use_contextual=not args.lexical_only)
    print(data.groupby(['source_type', 'training']).size().to_string())
