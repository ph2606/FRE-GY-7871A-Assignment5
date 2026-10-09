"""Aligned descriptive associations, HAC uncertainty and fixed-family tests."""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as student_t
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from .config import DATA, OUT, START, TRAIN_END, END, SEED, TOPIC_LABELS
from .text_model import aggregate_indices


def correlation_test(x, y, bootstrap=999, hac_lags=3):
    """Full-grid resampling keeps missing periods in place before pair deletion."""
    pair = pd.concat([x.rename('x'), y.rename('y')], axis=1).replace([np.inf, -np.inf], np.nan)
    valid = pair.dropna()
    result = {'n': len(valid), 'pearson_r': np.nan, 'spearman_r': np.nan,
        'hac_slope': np.nan, 'hac_p': np.nan, 'block_ci_low': np.nan, 'block_ci_high': np.nan}
    if len(valid) < 4 or valid.x.std() < 1e-12 or valid.y.std() < 1e-12:
        return result
    result['pearson_r'] = float(valid.x.corr(valid.y))
    result['spearman_r'] = float(spearmanr(valid.x, valid.y).statistic)
    if len(valid) < 12:
        return result
    # OLS complete pairs; put missing residual scores at zero on the FULL grid,
    # so a three-period HAC lag never bridges a missing calendar/session period.
    design = np.column_stack([np.ones(len(valid)), valid.x.to_numpy()])
    response = valid.y.to_numpy()
    bread = np.linalg.inv(design.T @ design)
    beta = bread @ design.T @ response
    residual = response - design @ beta
    score_grid = np.zeros((len(pair), 2))
    score_grid[pair.notna().all(axis=1).to_numpy()] = design * residual[:, None]
    meat = score_grid.T @ score_grid
    for lag in range(1, hac_lags + 1):
        cross = score_grid[lag:].T @ score_grid[:-lag]
        meat += (1 - lag / (hac_lags + 1)) * (cross + cross.T)
    covariance = bread @ meat @ bread * len(valid) / (len(valid) - 2)
    standard_error = np.sqrt(max(float(covariance[1, 1]), 0.))
    result['hac_slope'] = float(beta[1])
    result['hac_p'] = float(2 * student_t.sf(abs(beta[1] / standard_error), len(valid) - 2)) if standard_error > 0 else np.nan
    rng = np.random.default_rng(SEED)
    n = len(pair)
    correlations = []
    for _ in range(bootstrap):
        starts = rng.integers(0, n, size=int(np.ceil(n / 5)))
        positions = np.concatenate([(s + np.arange(5)) % n for s in starts])[:n]
        sampled = pair.iloc[positions].dropna()
        if len(sampled) >= 8 and sampled.x.std() > 1e-12 and sampled.y.std() > 1e-12:
            correlations.append(sampled.x.corr(sampled.y))
    if len(correlations) >= 100:
        result['block_ci_low'], result['block_ci_high'] = map(float, np.quantile(correlations, [.025, .975]))
    return result


def market_adjusted_returns(prices, tickers, benchmark='SPY'):
    """Equal daily weights require all current and preceding constituent prices."""
    selected = prices[tickers + [benchmark]]
    returns = selected.pct_change(fill_method=None)
    basket = returns[tickers].mean(axis=1, skipna=False)
    return pd.DataFrame({'basket': basket, 'benchmark': returns[benchmark], 'adjusted': basket - returns[benchmark]})


def prepare_panels(documents):
    probabilities = pd.read_csv(DATA / 'prediction_market_daily.csv')
    probabilities['date'] = probabilities.date.astype(str).str[:10]
    probabilities = probabilities.set_index('date')
    date_grid = pd.date_range(START, END, freq='D')
    calendar_closes = (date_grid.tz_localize('America/New_York') + pd.Timedelta(hours=16)).tz_convert('UTC')
    calendar = aggregate_indices(documents, calendar_closes)
    calendar['date'] = date_grid.strftime('%Y-%m-%d')
    calendar = calendar.set_index('date')
    for column in probabilities:
        if 'probability' in column:
            calendar[column] = pd.to_numeric(probabilities[column], errors='coerce').reindex(calendar.index)
    baskets = pd.read_csv(DATA / 'basket_daily.csv')
    baskets['date'] = baskets.date.astype(str).str[:10]
    baskets = baskets.set_index('date').sort_index()
    baskets = baskets.loc[(baskets.index >= START) & (baskets.index <= END)]
    stock_dates = pd.DatetimeIndex(pd.to_datetime(baskets.index))
    market_closes = (stock_dates.tz_localize('America/New_York') + pd.Timedelta(hours=16)).tz_convert('UTC')
    market = aggregate_indices(documents, market_closes)
    market['date'] = stock_dates.strftime('%Y-%m-%d')
    market = market.set_index('date').join(baskets)
    calendar.to_csv(OUT / 'calendar_panel.csv')
    market.to_csv(OUT / 'market_panel.csv')
    return calendar, market


def analyze():
    documents = pd.read_csv(DATA / 'scored_texts.csv', low_memory=False)
    for col in ['available_at', 'published_at']:
        documents[col] = pd.to_datetime(documents[col], utc=True, format='mixed')
    calendar, market = prepare_panels(documents)
    topics = [c for c in calendar if c.startswith('index_')]
    probability_targets = [c for c in calendar if c.endswith('_probability')]
    # Basket field names are stable contracts with the acquisition module.
    equity_targets = ['long_only_minus_spy', 'signed_beta_adjusted']
    missing = set(equity_targets) - set(market)
    if missing:
        raise ValueError(f'Basket acquisition is missing {missing}; inspect schema rather than guessing.')
    rows = []
    for panel, targets, frequency in [(calendar, probability_targets, 'calendar'), (market, equity_targets, 'trading')]:
        test_mask = panel.index > TRAIN_END
        for topic in topics:
            change = panel[topic].diff()
            for target in targets:
                outcome = panel[target].diff() if frequency == 'calendar' else panel[target]
                for lag in [0, 1]:
                    result = correlation_test(change.shift(lag).loc[test_mask], outcome.loc[test_mask])
                    rows.append({'frequency': frequency, 'index': topic, 'target': target,
                        'lag': lag, 'hypothesis': 'contemporaneous association' if lag == 0 else 'lagged association', **result})
    results = pd.DataFrame(rows)
    results['holm_p'] = np.nan
    usable = results.hac_p.notna()
    adjusted = multipletests(results.hac_p.fillna(1.), method='holm')[1]
    results.loc[usable, 'holm_p'] = adjusted[usable]
    results.to_csv(OUT / 'correlations.csv', index=False)
    flagged = []
    for _, selected in results.loc[results.holm_p.lt(.05)].iterrows():
        panel = calendar if selected.frequency == 'calendar' else market
        mask = panel.index > TRAIN_END
        x = panel[selected['index']].diff().shift(int(selected.lag)).loc[mask]
        y = (panel[selected.target].diff() if selected.frequency == 'calendar' else panel[selected.target]).loc[mask]
        for lags in [1, 3, 5]:
            flagged.append({'index': selected['index'], 'target': selected.target,
                'diagnostic': f'HAC_lags_{lags}', **correlation_test(x, y, bootstrap=0, hac_lags=lags)})
        paired = pd.concat([x.rename('x'), y.rename('y')], axis=1).dropna()
        r = selected.pearson_r
        ordinary_p = float(2 * student_t.sf(abs(r * np.sqrt((len(paired)-2)/(1-r*r))), len(paired)-2))
        leave_r = []
        for date in paired.index:
            dropped_x = x.copy()
            dropped_x.loc[date] = np.nan
            leave_r.append(correlation_test(dropped_x, y, bootstrap=0)['pearson_r'])
        flagged.append({'index': selected['index'], 'target': selected.target,
            'diagnostic': 'ordinary_iid_and_leave_one_pair_out', 'n': len(paired), 'pearson_r': r,
            'ordinary_iid_p': ordinary_p, 'leave_one_r_min': min(leave_r), 'leave_one_r_max': max(leave_r)})
    pd.DataFrame(flagged, columns=['index','target','diagnostic','n','pearson_r','hac_p','ordinary_iid_p','leave_one_r_min','leave_one_r_max']).to_csv(OUT / 'flagged_diagnostics.csv', index=False)
    # Levels are an explicitly separate, nonstationary descriptive exhibit.
    levels = []
    for target in probability_targets:
        result = correlation_test(calendar.index_all.loc[calendar.index > TRAIN_END], calendar[target].loc[calendar.index > TRAIN_END], bootstrap=0)
        levels.append({'index': 'index_all', 'target': target, 'n': result['n'], 'pearson_r': result['pearson_r'], 'warning': 'Levels can be spurious; primary tests use changes.'})
    pd.DataFrame(levels).to_csv(OUT / 'level_correlations.csv', index=False)
    # Sensitivities all use the broad index and same primary market targets.
    sensitivities = []
    for name, kwargs in [('equal_author_outlet', {'equal_outlets': True}), ('vader_lexical', {'score': 'vader'}), ('polarity_balance', {'score': 'polarity_balance'})]:
        closes = (pd.DatetimeIndex(pd.to_datetime(calendar.index)).tz_localize('America/New_York') + pd.Timedelta(hours=16)).tz_convert('UTC')
        alt = aggregate_indices(documents, closes, **kwargs)
        alt.index = calendar.index
        for target in probability_targets:
            result = correlation_test(alt.index_all.diff().loc[calendar.index > TRAIN_END], calendar[target].diff().loc[calendar.index > TRAIN_END], bootstrap=0)
            sensitivities.append({'specification': name, 'target': target, **result})
    for source in ['news', 'social']:
        for target in probability_targets:
            result = correlation_test(calendar[f'{source}_all'].diff().loc[calendar.index > TRAIN_END], calendar[target].diff().loc[calendar.index > TRAIN_END], bootstrap=0)
            sensitivities.append({'specification': source + '_only', 'target': target, **result})
    pd.DataFrame(sensitivities).to_csv(OUT / 'sensitivities.csv', index=False)
    # Equity-level results are exploratory; adjust their separate fixed family.
    equities = pd.read_csv(DATA / 'equity_returns.csv')
    equities['date'] = equities.date.astype(str).str[:10]
    equity_rows = []
    stock_mask = market.index > TRAIN_END
    for ticker, group in equities.groupby('ticker'):
        group = group.set_index('date').reindex(market.index)
        for adjustment in ['market_adjusted_return_pct', 'beta_adjusted_return_pct', 'sector_adjusted_return_pct']:
            result = correlation_test(market.index_all.diff().loc[stock_mask], (group[adjustment] / 100).loc[stock_mask], bootstrap=0)
            equity_rows.append({'ticker': ticker, 'adjustment': adjustment, **result})
    equity_table = pd.DataFrame(equity_rows)
    equity_table['holm_p'] = np.nan
    ok = equity_table.hac_p.notna()
    adjusted = multipletests(equity_table.hac_p.fillna(1.), method='holm')[1]
    equity_table.loc[ok, 'holm_p'] = adjusted[ok]
    equity_table.to_csv(OUT / 'individual_equity_correlations.csv', index=False)
    outcome_rows = []
    for target in probability_targets:
        pchange = calendar[target].reindex(market.index).diff()
        for equity_target in equity_targets:
            result = correlation_test(pchange.loc[stock_mask], market[equity_target].loc[stock_mask], bootstrap=0)
            outcome_rows.append({'probability': target, 'equity': equity_target, **result})
    pd.DataFrame(outcome_rows).to_csv(OUT / 'probability_equity_associations.csv', index=False)
    performance = []
    for ticker, group in equities.groupby('ticker'):
        group = group.set_index('date').sort_index()
        group = group.loc[group.index > TRAIN_END]
        performance.append({'ticker': ticker, 'test_n': len(group),
            'total_return_pct': float(((1 + group.return_pct / 100).prod() - 1) * 100),
            'sum_spy_adjusted_pct': float(group.market_adjusted_return_pct.sum()),
            'sum_beta_adjusted_pct': float(group.beta_adjusted_return_pct.sum()),
            'sum_sector_adjusted_pct': float(group.sector_adjusted_return_pct.sum())})
    pd.DataFrame(performance).to_csv(OUT / 'equity_performance.csv', index=False)
    terms = pd.read_csv(OUT / 'topic_terms.csv')
    summaries = []
    for _, topic in terms.iterrows():
        key = topic.topic
        record = topic.to_dict()
        record['display_label'] = TOPIC_LABELS.get(key, topic.label)
        for source in ['news', 'social']:
            selected = documents.loc[documents.source_type.eq(source)]
            record[source + '_share'] = float(selected[key].mean())
            record[source + '_valence'] = float(np.average(selected.sentiment, weights=selected[key]))
        summaries.append(record)
    pd.DataFrame(summaries).to_csv(OUT / 'topic_summary.csv', index=False)
    coverage = calendar[['news_n', 'social_n', 'index_all']].copy()
    coverage.to_csv(OUT / 'daily_coverage.csv')
    summary = {'requested_start': START, 'train_end': TRAIN_END, 'end': END,
        'documents': documents.groupby('source_type').size().astype(int).to_dict(),
        'training_documents': int(documents.training.sum()), 'model_covered': int(documents.model_covered.sum()),
        'topic_k': len(terms), 'calendar_days': len(calendar), 'market_sessions': len(market),
        'test_calendar_days': int((calendar.index > TRAIN_END).sum()),
        'test_market_sessions': int((market.index > TRAIN_END).sum()),
        'index_days_present': int(calendar.index_all.notna().sum()),
        'core_family_size': len(results), 'tested_hypotheses': int(usable.sum()), 'holm_significant': int(results.holm_p.lt(.05).sum()),
        'last_polymarket_probability': float(calendar.polymarket_probability.dropna().iloc[-1]),
        'inference': 'Associations, not causality. HAC maxlags=3, circular 5-period block bootstrap; small samples and gaps limit inference.'}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    return calendar, market, results


if __name__ == '__main__':
    analyze()
