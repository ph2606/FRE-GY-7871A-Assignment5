"""Small synthetic checks for timing, missingness and weighting conventions.

These fixtures validate software only; none supplies empirical observations.
"""
import numpy as np
import pandas as pd
import pytest
import re
from types import SimpleNamespace
from scipy.sparse import csr_matrix
import statsmodels.api as sm

from src.text_model import (
    aggregate_indices, align_to_closes, normalize_loadings, npmi_score,
)
from src.analysis import correlation_test, market_adjusted_returns
from src.market_data import daily_probability, nyse_sessions
from src.contextual_sentiment import preprocess, token_chunks
from src.issue_sentiment import ISSUES
from src import collect_text


@pytest.fixture
def closes():
    return pd.DatetimeIndex([
        '2026-08-21T20:00:00Z',  # Friday, 16:00 New York
        '2026-08-24T20:00:00Z',  # Monday
        '2026-08-25T20:00:00Z',
    ])


def document(identifier, kind, timestamp, sentiment, source='Outlet A', author=''):
    return dict(id=identifier, source_type=kind, available_at=timestamp,
                sentiment=sentiment, source=source, author=author,
                topic_0=.8, topic_1=.2, model_covered=True)


def test_nmf_loadings_preserve_mass_and_leave_unrecognized_text_unclassified():
    actual = normalize_loadings([[2., 6.], [0., 0.], [20., 60.]])
    np.testing.assert_allclose(actual, [[.25, .75], [0., 0.], [.25, .75]])
    np.testing.assert_allclose(actual.sum(axis=1), [1., 0., 1.])


def test_after_close_and_weekend_text_enters_next_actual_session(closes):
    observations = pd.Series([
        '2026-08-21T19:59:59Z', '2026-08-21T20:00:00Z',
        '2026-08-21T20:00:01Z', '2026-08-23T12:00:00Z',
        '2026-08-25T20:00:01Z', None,
    ], index=[10, 20, 30, 40, 50, 60])
    actual = align_to_closes(observations, closes)
    assert actual.index.equals(observations.index)
    assert actual.iloc[:4].tolist() == [closes[0], closes[0], closes[1], closes[1]]
    assert actual.iloc[4:].isna().all()


def test_market_closes_handle_daylight_saving_change():
    closes = pd.DatetimeIndex(['2026-10-30T20:00:00Z', '2026-11-02T21:00:00Z'])
    actual = align_to_closes(pd.Series(['2026-11-02T20:30:00Z']), closes)
    assert actual.iloc[0] == closes[1]


@pytest.mark.parametrize('invalid_closes', [
    ['2026-08-24T20:00:00Z', '2026-08-21T20:00:00Z'],
    ['2026-08-21T20:00:00Z', '2026-08-21T20:00:00Z'],
])
def test_invalid_close_calendars_are_rejected(invalid_closes):
    with pytest.raises(ValueError, match='sorted and unique'):
        align_to_closes(pd.Series(['2026-08-21T19:00:00Z']), invalid_closes)


def test_missing_medium_never_becomes_neutral_or_a_compressed_session(closes):
    records = pd.DataFrame([
        document('n1', 'news', '2026-08-21T18:00:00Z', -.8),
        document('s1', 'social', '2026-08-21T18:00:00Z', .4),
        document('n2', 'news', '2026-08-25T18:00:00Z', .5),
    ])
    actual = aggregate_indices(records, closes)
    assert actual.index.equals(closes.rename('close'))
    assert actual.loc[closes[0], 'index_all'] == pytest.approx(-.2)
    assert pd.isna(actual.loc[closes[1], 'index_all'])
    assert pd.isna(actual.loc[closes[2], 'index_all'])
    assert actual.loc[closes[2], 'news_all'] == .5
    # Topic 1 has insufficient mass in each medium, so its score is missing.
    assert actual['index_topic_1'].isna().all()


def test_zero_topic_loading_does_not_add_an_unrelated_document(closes):
    records = pd.DataFrame([
        document('n1', 'news', '2026-08-21T18:00:00Z', -.8),
        document('s1', 'social', '2026-08-21T18:00:00Z', .4),
        document('oov', 'news', '2026-08-21T18:00:00Z', 1.),
    ])
    records.loc[records.id.eq('oov'), ['topic_0', 'topic_1']] = 0.
    records.loc[records.id.eq('oov'), 'model_covered'] = False
    actual = aggregate_indices(records, closes)
    assert actual.loc[closes[0], 'news_n'] == 1
    assert actual.loc[closes[0], 'index_all'] == pytest.approx(-.2)


def test_equal_entity_sensitivity_removes_repeated_author_and_outlet_dominance(closes):
    records = []
    for kind in ['news', 'social']:
        for i in range(3):
            records.append(document(f'{kind}_a{i}', kind, '2026-08-21T18:00:00Z',
                                    -.8, source='Outlet A', author='author_a'))
        records.append(document(f'{kind}_b', kind, '2026-08-21T18:00:00Z',
                                .8, source='Outlet B', author='author_b'))
    records = pd.DataFrame(records)
    standard = aggregate_indices(records, closes)
    sensitivity = aggregate_indices(records, closes, equal_outlets=True)
    assert standard.loc[closes[0], 'index_all'] == pytest.approx(-.4)
    assert sensitivity.loc[closes[0], 'index_all'] == pytest.approx(0.)


@pytest.mark.parametrize('observed, expected', [
    ([[1, 1], [1, 1], [0, 0], [0, 0]], 1.),
    ([[1, 1], [1, 0], [0, 1], [0, 0]], 0.),
    ([[1, 0], [1, 0], [0, 1], [0, 1]], -1.),
])
def test_topic_coherence_has_known_cooccurrence_bounds(observed, expected):
    score = npmi_score(csr_matrix(observed), np.array([[1., 1.]]), top_n=2)
    assert score == pytest.approx(expected)


def test_prediction_cutoff_includes_equality_and_excludes_later_quotes():
    history = pd.DataFrame({
        'platform': ['polymarket'] * 3,
        'date_time': pd.to_datetime([
            '2026-08-01T19:59:00Z', '2026-08-01T20:00:00Z',
            '2026-08-01T20:00:01Z',
        ], utc=True).tz_convert('America/New_York'),
        'probability': [.2, .4, .9],
    })
    actual = daily_probability(history, '2026-08-01', '2026-08-02')
    assert actual.polymarket_probability.tolist() == [.4, .9]
    assert actual.polymarket_age_hours.iloc[0] == 0.
    assert actual.polymarket_observation_at.iloc[0] <= actual.cutoff_at.iloc[0]


def test_prediction_prices_over_24_hours_old_stay_missing():
    history = pd.DataFrame({
        'platform': ['polymarket', 'kalshi'],
        'date_time': pd.to_datetime([
            '2026-07-31T19:59:00Z', '2026-07-31T20:00:00Z',
        ], utc=True).tz_convert('America/New_York'),
        'probability': [.2, .3],
    })
    actual = daily_probability(history, '2026-08-01', '2026-08-02')
    assert pd.isna(actual.polymarket_probability.iloc[0])
    assert actual.kalshi_probability.iloc[0] == .3  # Exactly 24 hours old.
    assert actual[['polymarket_probability', 'kalshi_probability']].iloc[1].isna().all()


def test_missing_constituent_price_never_reweights_or_bridges_a_return():
    prices = pd.DataFrame({
        'A': [100., 110., 121., 133.1],
        'B': [100., np.nan, 110., 121.],
        'SPY': [100., 101., 102.01, 103.0301],
    }, index=pd.date_range('2026-08-03', periods=4, freq='B'))
    actual = market_adjusted_returns(prices, ['A', 'B'])
    assert actual.basket.iloc[:3].isna().all()
    assert actual.adjusted.iloc[3] == pytest.approx(.09)


def test_cash_calendar_excludes_labor_day_and_keeps_next_session():
    actual = nyse_sessions('2026-09-04', '2026-09-09')
    assert actual.strftime('%Y-%m-%d').tolist() == [
        '2026-09-04', '2026-09-08', '2026-09-09',
    ]


def test_correlations_join_dates_and_report_insufficient_pairs():
    x = pd.Series([1., 2., 3.], index=['a', 'b', 'c'])
    y = pd.Series([9., 6., 3.], index=['b', 'c', 'd'])
    actual = correlation_test(x, y, bootstrap=0)
    assert actual['n'] == 2
    assert np.isnan(actual['pearson_r'])
    assert np.isnan(actual['hac_p'])


def test_constant_series_does_not_produce_a_significance_claim():
    actual = correlation_test(pd.Series([1.] * 20), pd.Series(range(20)), bootstrap=0)
    assert actual['n'] == 20
    assert np.isnan(actual['pearson_r'])
    assert np.isnan(actual['hac_p'])


def test_short_sample_can_have_descriptive_correlation_without_hac_test():
    x = pd.Series([0., 1., 2., 3., 4., 5.])
    actual = correlation_test(x, 2 * x, bootstrap=0)
    assert actual['n'] == 6
    assert actual['pearson_r'] == pytest.approx(1.)
    assert np.isnan(actual['hac_p'])
    assert np.isnan(actual['block_ci_low'])


def test_unit_conversion_changes_slope_and_preserves_correlation():
    x = pd.Series(np.linspace(-.5, .5, 20))
    disturbance = pd.Series(np.sin(np.arange(20)) * .001)
    y = .02 * x + disturbance
    decimals = correlation_test(x, y, bootstrap=0)
    percentages = correlation_test(x, 100 * y, bootstrap=0)
    assert decimals['n'] == percentages['n'] == 20
    assert percentages['pearson_r'] == pytest.approx(decimals['pearson_r'])
    assert percentages['hac_slope'] == pytest.approx(100 * decimals['hac_slope'])


def test_hac_matches_statsmodels_on_a_complete_time_grid():
    rng = np.random.default_rng(787105)
    x = pd.Series(rng.normal(size=60), name='x')
    shocks = rng.normal(size=60)
    residual = np.zeros(60)
    for i in range(1, len(residual)):
        residual[i] = .6 * residual[i - 1] + shocks[i]
    y = .3 * x + pd.Series(residual)
    reference = sm.OLS(y, sm.add_constant(x)).fit(
        cov_type='HAC', cov_kwds={'maxlags': 3, 'use_correction': True},
        use_t=True,
    )
    actual = correlation_test(x, y, bootstrap=0)
    assert actual['hac_slope'] == pytest.approx(reference.params['x'], rel=1e-12)
    assert actual['hac_p'] == pytest.approx(reference.pvalues['x'], rel=1e-10)


def test_hac_preserves_gaps_instead_of_treating_distant_scores_as_adjacent():
    rng = np.random.default_rng(787113)
    x = pd.Series(rng.normal(size=60), name='x')
    shocks = rng.normal(size=60)
    residual = np.zeros(60)
    for i in range(1, len(residual)):
        residual[i] = .8 * residual[i - 1] + shocks[i]
    y = .3 * x + pd.Series(residual)
    # Missing sessions retain their positions in the HAC lag calendar.
    gaps = [8, 9, 10, 11, 21, 22, 23, 24, 38, 39, 40, 41]
    x.loc[gaps] = np.nan
    y.loc[gaps] = np.nan
    observed = pd.concat([x.rename('x'), y.rename('y')], axis=1).dropna()
    compressed = sm.OLS(observed.y, sm.add_constant(observed.x)).fit(
        cov_type='HAC', cov_kwds={'maxlags': 3, 'use_correction': True},
        use_t=True,
    )
    actual = correlation_test(x, y, bootstrap=0)
    assert actual['n'] == len(observed)
    # Gaps affect covariance, not the complete-pair OLS slope.
    assert actual['hac_slope'] == pytest.approx(compressed.params['x'], rel=1e-12)
    assert abs(actual['hac_p'] - compressed.pvalues['x']) > 1e-4


@pytest.mark.parametrize('length', [510, 511, 1200])
def test_contextual_chunks_retain_every_payload_token_in_order(length):
    original = list(range(length))
    chunks = token_chunks(original)
    assert [token for chunk in chunks for token in chunk] == original
    assert all(0 < len(chunk) <= 510 for chunk in chunks)


def test_contextual_social_normalization_preserves_ordinary_text_and_email():
    original = '@Some_User Read https://example.org/a?q=b mail me@example.org now.'
    assert preprocess(original) == '@user Read http mail me@example.org now.'


def test_weighted_topic_coherence_uses_document_weights_and_ignores_their_scale():
    observed = csr_matrix([[1, 1], [1, 0], [0, 1], [0, 0]])
    weights = np.array([4., 1., 1., 4.])
    # Each term has prevalence .5 and joint weighted prevalence .4.
    expected = np.log(.4 / (.5 * .5)) / -np.log(.4)
    actual = npmi_score(observed, np.array([[1., 1.]]), top_n=2,
                        sample_weight=weights)
    scaled = npmi_score(observed, np.array([[1., 1.]]), top_n=2,
                        sample_weight=weights * 100)
    assert actual == pytest.approx(expected)
    assert scaled == pytest.approx(expected)


@pytest.fixture
def screen_english(monkeypatch):
    # Isolate relevance rules from a probabilistic language-detector decision.
    monkeypatch.setattr(collect_text, 'detect_langs',
                        lambda _: [SimpleNamespace(lang='en', prob=1.)])
    def screen(text, medium='social', story_title=''):
        row = dict(text=text, source_type=medium, story_title=story_title,
                   published_at='2026-08-10T12:00:00Z')
        return collect_text.screen(
            row, pd.Timestamp('2026-08-01', tz='America/New_York').to_pydatetime(),
            pd.Timestamp('2026-10-09', tz='America/New_York').to_pydatetime(),
        )
    return screen


def test_country_screen_distinguishes_the_pronoun_us_from_uppercase_us(screen_english):
    assert screen_english('Voters tell us about their next election.', 'news') == (
        'no explicit US political context')
    assert screen_english('US voters discuss the next election.', 'news') == ''


def test_a_year_only_in_a_citation_url_does_not_establish_current_midterms(screen_english):
    historical = ('Trump won the 2024 election despite polling errors described in '
                  'https://example.org/2026/analysis')
    assert screen_english(historical) == (
        'no explicit dated-election or congressional-midterm context')
    current = ('Trump won the 2024 election, but voters now discuss '
               'the upcoming November 2026 midterm elections.')
    assert screen_english(current) == ''


def test_union_ballots_need_an_explicit_congressional_link(screen_english):
    union = 'The US union election this November may change factory bargaining plans.'
    assert screen_english(union) == (
        'nonpolitical election without explicit congressional-midterm context')
    linked = ('The US union election organizers are also campaigning '
              'in the congressional midterm elections this November.')
    assert screen_english(linked) == ''


@pytest.mark.parametrize('issue, unrelated, related', [
    ('education', 'The University of New Hampshire poll shows a close Senate race.',
     'Voters want more funding for public schools and teachers.'),
    ('healthcare', 'Candidates increased spending on premium CTV advertising.',
     'Voters want lower health insurance premiums in November.'),
    ('foreign_policy', 'Republicans build a campaign war chest for the Senate election.',
     'Voters criticized the Iran war and military spending.'),
])
def test_issue_rules_distinguish_pollsters_advertising_and_campaign_metaphors(
        issue, unrelated, related):
    pattern = re.compile(ISSUES[issue], re.I)
    assert pattern.search(unrelated) is None
    assert pattern.search(related) is not None
