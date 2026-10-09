"""Auditable policy-topic sensitivity alongside unsupervised NMF topics.

This fixed, overlapping dictionary is a supplementary taxonomy, not another
unsupervised model. It tests named voter issues that campaign clusters can miss.
Contextual valence uses issue-bearing records; sentence-restricted lexical
valence provides a target-scope sensitivity. Quotations/stance remain limits.
No election price or equity return defines these patterns.
"""
import re
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from .config import DATA, OUT, TRAIN_END, START, END
from .text_model import aggregate_indices
from .analysis import correlation_test

ISSUES = {
    'economy': r'\b(?:econom\w*|inflation|afford\w*|cost of living|prices?|tax(?:es|ation)?|tariffs?|jobs?|wages?|unemployment|housing|rents?|capital gains)\b',
    'democracy_ethics': r'\b(?:democra(?:cy|cies)|authoritarian\w*|dictator\w*|voting rights|voter suppress\w*|election (?:fraud|interfer\w*|integrity)|rig(?:ging|ged)?|gerrymander\w*|corrupt\w*|ethics|constitutional|oligarch\w*)\b',
    'healthcare': r'\b(?:health\s?care|health insurance|medicare|medicaid|obamacare|affordable care|hospitals?|drug prices?|insurance premiums?|health premiums?)\b',
    'immigration': r'\b(?:immigra\w*|deport\w*|border|migrants?|asylum|detention|ICE)\b',
    'foreign_policy': r'\b(?:foreign policy|warfare|armed conflict|Iran|Israel|Gaza|Ukraine|Russia|China|NATO|military|national defen[cs]e|defen[cs]e spending|troops?|Pentagon|missiles?|bombing|ceasefire)\b',
    'energy_climate': r'\b(?:energy|climate|solar|oil|gas(?:oline)?|fossil|renewable\w*|carbon|environment\w*|emissions?)\b',
    'education': r'\b(?:education|schools?|teachers?|tuition|student (?:loans?|debt))\b',
    'crime_guns': r'\b(?:violent crime|crime rates?|police (?:force|officers?|departments?)|law enforcement|guns?|shootings?|firearms?|public safety|violence)\b',
    'abortion': r'\b(?:abortion|reproductive|Roe|pro.choice|pro.life)\b',
}


def build():
    docs = pd.read_csv(DATA / 'scored_texts.csv', low_memory=False)
    docs['available_at'] = pd.to_datetime(docs.available_at, utc=True, format='mixed')
    analyzer = SentimentIntensityAnalyzer()
    patterns = {name: re.compile(pattern, re.I if name != 'immigration' else 0) for name, pattern in ISSUES.items()}
    # ICE is uppercase; other immigration stems should still match case variants.
    patterns['immigration'] = re.compile(r'(?i:\b(?:immigra\w*|deport\w*|border|migrants?|asylum|detention)\b)|\bICE\b')
    for name, pattern in patterns.items():
        matching = []
        for text in docs.analysis_text:
            sentences = re.split(r'(?<=[.!?])\s+', str(text))
            selected = [s for s in sentences if pattern.search(s)]
            matching.append(np.mean([analyzer.polarity_scores(s)['compound'] for s in selected]) if selected else np.nan)
        docs[name + '_valence'] = matching
    docs.to_csv(DATA / 'issue_scored_texts.csv', index=False)
    summary, rows = [], []
    calendar = pd.read_csv(OUT / 'calendar_panel.csv', index_col='date')
    market = pd.read_csv(OUT / 'market_panel.csv', index_col='date')
    dates = pd.date_range(START, END)
    closes = (dates.tz_localize('America/New_York') + pd.Timedelta(hours=16)).tz_convert('UTC')
    market_closes = (pd.DatetimeIndex(pd.to_datetime(market.index)).tz_localize('America/New_York') + pd.Timedelta(hours=16)).tz_convert('UTC')
    daily = pd.DataFrame(index=calendar.index)
    for name in ISSUES:
        relevant = docs.loc[docs[name + '_valence'].notna()].copy()
        relevant['sentence_vader'] = relevant[name + '_valence']
        relevant['topic_0'] = 1.
        relevant['model_covered'] = True
        relevant = relevant.drop(columns=[c for c in relevant if re.fullmatch(r'topic_[1-9]\d*', c)])
        indices = aggregate_indices(relevant, closes)
        indices.index = calendar.index
        daily[name] = indices.index_all
        for source in ['news', 'social']:
            selected = relevant.loc[relevant.source_type.eq(source)]
            summary.append({'issue': name, 'source_type': source, 'n': len(selected),
                'share_pct': len(selected) / (docs.source_type.eq(source).sum()) * 100,
                'contextual_document_valence': selected.sentiment.mean(),
                'sentence_vader_valence': selected.sentence_vader.mean()})
        mask = calendar.index > TRAIN_END
        for target in ['polymarket_probability', 'kalshi_probability']:
            for specification in ['equal_media', 'news_only']:
                x = indices.index_all if specification == 'equal_media' else indices.news_all
                result = correlation_test(x.diff().loc[mask], calendar[target].diff().loc[mask], bootstrap=0)
                rows.append({'issue': name, 'specification': specification, 'target': target, **result})
        market_indices = aggregate_indices(relevant, market_closes)
        market_indices.index = market.index
        market_mask = market.index > TRAIN_END
        for target in ['long_only_minus_spy', 'signed_beta_adjusted']:
            for specification in ['equal_media', 'news_only']:
                x = market_indices.index_all if specification == 'equal_media' else market_indices.news_all
                result = correlation_test(x.diff().loc[market_mask], market[target].loc[market_mask], bootstrap=0)
                rows.append({'issue': name, 'specification': specification, 'target': target, **result})
        lexical = aggregate_indices(relevant, closes, score='sentence_vader')
        lexical.index = calendar.index
        for target in ['polymarket_probability', 'kalshi_probability']:
            result = correlation_test(lexical.news_all.diff().loc[mask], calendar[target].diff().loc[mask], bootstrap=0)
            rows.append({'issue': name, 'specification': 'sentence_vader_news', 'target': target, **result})
    pd.DataFrame(summary).to_csv(OUT / 'issue_summary.csv', index=False)
    table = pd.DataFrame(rows)
    from statsmodels.stats.multitest import multipletests
    table['holm_p'] = np.nan
    usable = table.hac_p.notna()
    adjusted = multipletests(table.hac_p.fillna(1.), method='holm')[1]
    table.loc[usable, 'holm_p'] = adjusted[usable]
    table.to_csv(OUT / 'issue_correlations.csv', index=False)
    daily.to_csv(OUT / 'issue_daily_indices.csv')
    return table


if __name__ == '__main__':
    build()
