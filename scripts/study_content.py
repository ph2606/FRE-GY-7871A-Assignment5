"""One computed narrative shared by the six-page PDF and notebook."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.config import TOPIC_LABELS

ROOT = Path(__file__).resolve().parents[1]
REPO = 'https://github.com/ph2606/FRE-GY-7871A-Assignment5'
PEW = 'https://www.pewresearch.org/politics/2026/10/08/economy-ranks-as-top-issue-in-midterms-for-all-voters-but-partisans-differ-over-the-importance-of-other-issues/'
JULY = 'https://www.pewresearch.org/politics/2026/07/23/as-the-2026-midterms-approach-economy-is-front-and-center/'
LABELS = TOPIC_LABELS


def read(name):
    return pd.read_csv(ROOT / 'outputs' / (name + '.csv'))


def p(text):
    return {'type': 'p', 'text': text}


def table(headers, rows, widths=None):
    return {'type': 'table', 'headers': headers, 'rows': rows, 'widths': widths}


def fig(name, caption, width='.98'):
    return {'type': 'figure', 'name': name, 'caption': caption, 'width': width}


def f(value, digits=2):
    return '—' if pd.isna(value) else f'{value:.{digits}f}'


def sections():
    summary = json.loads((ROOT / 'outputs/summary.json').read_text())
    method = json.loads((ROOT / 'outputs/text_method.json').read_text())
    selection = read('topic_model_selection')
    topics = read('topic_summary')
    topics['display_label'] = topics.topic.map(LABELS).fillna(topics.label)
    topics.to_csv(ROOT / 'outputs/topic_summary.csv', index=False)
    core = read('correlations')
    broad = core.loc[core['index'].eq('index_all')]
    perf = read('equity_performance')
    market = read('market_panel')
    cal = read('calendar_panel')
    policy = read('issue_summary')
    issues = read('issue_correlations')
    ai = json.loads((ROOT / 'outputs/sentiment_review_summary.json').read_text()) if (ROOT / 'outputs/sentiment_review_summary.json').exists() else {}
    contextual = json.loads((ROOT / 'outputs/contextual_review_summary.json').read_text()) if (ROOT / 'outputs/contextual_review_summary.json').exists() else {}
    if contextual:
        audit_n = contextual['n']
        neural = contextual['contextual']
        lexical = contextual['vader']
        positive_n = sum(contextual['contextual_confusion'][2])
        positive_correct = contextual['contextual_confusion'][2][2]
        audit_text = (f'A frozen, stratified 60-text AI-reference diagnostic was labeled before model scores were viewed. '
            f'{audit_n} records remain eligible after final timing/relevance screens. On the same records, contextual agreement is '
            f'{neural["agreements"]}/{audit_n} ({neural["ai_reference_agreement"]*100:.1f}%), compared with VADER '
            f'{lexical["agreements"]}/{audit_n} ({lexical["ai_reference_agreement"]*100:.1f}%). Contextual macro-F1 is '
            f'{neural["ai_reference_macro_f1"]:.3f}; positive-class support is only {positive_n}, with '
            f'{positive_correct} correctly classified. This development check informed model comparison; it is neither '
            f'an untouched test nor independent human validation. Sarcasm, quotation and target ambiguity remain. '
            f'Wording valence does not establish party support, issue approval or calibrated voter sentiment.')
    else:
        audit_text = ('Detailed blind-reference labels are unavailable in this checkout, so no fresh agreement estimate is reported. '
            'The submitted notebook retains the original aggregate AI-reference development diagnostic. '
            'Neither model has independent human validation; wording valence is not party support or voter intention.')
    model_name = 'contextual RoBERTa' if 'Cardiff' in method['sentiment'] else 'VADER'
    def result(target, lag=0):
        return broad.loc[broad.target.eq(target) & broad.lag.eq(lag)].iloc[0]
    poly = result('polymarket_probability')
    kalshi = result('kalshi_probability')
    basket = result('long_only_minus_spy')
    signed = result('signed_beta_adjusted')
    basket_return = ((1 + market.long_only_return_pct / 100).prod() - 1) * 100
    signed_return = ((1 + market.signed_return_pct / 100).prod() - 1) * 100
    spy_return = ((1 + market.spy_return_pct / 100).prod() - 1) * 100
    winner = selection.loc[selection.selected].iloc[0]
    significant = core.loc[core.holm_p.lt(.05)]
    if significant.empty:
        inference = f'None of the {summary["core_family_size"]} core associations survives Holm correction at 5%.'
    else:
        outcome_names = {'signed_beta_adjusted': 'signed beta-adjusted basket', 'long_only_minus_spy': 'five-stock basket minus SPY',
            'polymarket_probability': 'Polymarket quote change', 'kalshi_probability': 'Kalshi quote change'}
        entries = '; '.join(f'{TOPIC_LABELS.get(r["index"].removeprefix("index_"), "Broad wording")} and {outcome_names[r.target]}, lag {r.lag}: r={f(r.pearson_r)}, n={int(r.n)}, adjusted p={f(r.holm_p,3)}' for _, r in significant.iterrows())
        verb = 'survives' if len(significant) == 1 else 'survive'
        inference = f'{len(significant)} of {summary["core_family_size"]} core associations {verb} Holm correction: {entries}. Independent confirmation is needed.'
    flagged_path = ROOT / 'outputs/flagged_diagnostics.csv'
    diagnostic_text = ''
    if flagged_path.exists():
        flagged = pd.read_csv(flagged_path)
        ordinary = flagged.loc[flagged.diagnostic.eq('ordinary_iid_and_leave_one_pair_out')]
        if len(ordinary):
            check = ordinary.iloc[0]
            diagnostic_text = (f' For the flagged contemporaneous result, an ordinary IID test gives p={check.ordinary_iid_p:.3f}, '
                f'while leave-one-pair-out r ranges from {check.leave_one_r_min:.2f} to {check.leave_one_r_max:.2f}. '
                'Covariance assumptions matter, and contemporaneous association does not establish a forecast.')
    content = []
    content.append(('Question, evidence and main result', [
        p('I study the November 2026 U.S. midterms as a possible market event. A Democratic sweep means Democratic control of both the House and Senate. The presidency is not on the ballot. The research asks which issues appear in election discourse, whether the tone around those issues moves with sweep prices, and whether the same tone relates to a policy-exposed equity basket.'),
        p('The strongest evidence about voter priorities is the survey benchmark: the economy, the future of democracy and healthcare lead the October Pew ratings. The unsupervised text model mainly finds campaigning, Senate races, voting and political commentary. That difference matters. A ranked news feed and a technology forum show the subjects being discussed in this sample; they do not measure the electorate’s issue weights.'),
        table(['Study component', 'Observed design'], [
            ['Information window', 'Aug. 1–Oct. 8, 2026; final cutoff 16:00 New York'],
            ['Model training', f'Text available by Aug. 21 close; {summary["training_documents"]:,} eligible documents'],
            ['Assessment period', f'Aug. 22–Oct. 8; {summary["test_calendar_days"]} calendar days / {summary["test_market_sessions"]} cash sessions'],
            ['Aligned corpus', f'{summary["documents"]["news"]:,} news headlines + {summary["documents"]["social"]:,} HN records'],
            ['Prediction markets', 'Direct joint DD contracts on Polymarket and Kalshi'],
            ['Equities', 'FSLR, ENPH, RUN, GEO, CXW; UNH diagnostic only'],
        ], [.25, .75]),
        p(f'Over the complete window, Polymarket’s joint sweep quote rose from {cal.polymarket_probability.iloc[0]*100:.1f}% to {cal.polymarket_probability.iloc[-1]*100:.1f}%; Kalshi moved from {cal.kalshi_probability.iloc[0]*100:.1f}% to {cal.kalshi_probability.iloc[-1]*100:.1f}%. Yet the five-stock long-only basket returned {basket_return:+.2f}%, compared with SPY’s {spy_return:+.2f}%. The signed solar-long/detention-short basket returned {signed_return:+.2f}% before financing and trading costs.'),
        p(f'Using changes in the combined {model_name} wording index, the assessment-period same-day correlations with changes in the two contract quotes are {f(poly.pearson_r)} (Polymarket, n={int(poly.n)}) and {f(kalshi.pearson_r)} (Kalshi, n={int(kalshi.n)}). Its correlation with the five-stock basket minus SPY is {f(basket.pearson_r)} over {int(basket.n)} sessions. {inference}'),
        p('My investment conclusion is to use the issues and contracts for scenario monitoring, rather than place a sentiment-timing trade from this sample. The observed equity moves do not validate the simple claim that a rising sweep price automatically lifts the proposed beneficiaries.'),
    ]))
    poll_rows = []
    issue_map = [('economy', 'Economy', 84), ('democracy_ethics', 'Future of democracy', 73),
        ('healthcare', 'Healthcare', 68), ('education', 'Education', 62), ('immigration', 'Immigration', 59),
        ('foreign_policy', 'Foreign policy', 57), ('energy_climate', 'Energy policy', 54), ('crime_guns', 'Violent crime', 52)]
    for key, name, pct in issue_map:
        row = policy.loc[policy.issue.eq(key)].set_index('source_type')
        poll_rows.append([name, f'{pct}%', f'{row.loc["news","share_pct"]:.1f}%', f'{row.loc["social","share_pct"]:.1f}%'])
    content.append(('What voters care about, and what the model finds', [
        p('The collection uses broad election queries without sentiment words or return information. Google News RSS provides ranked, capped headlines from many named publishers. HN Algolia supplies comments and a small number of stories. Exact normalized duplicates are removed within each medium. Foreign, historical-election and language screens reduce irrelevant records but do not verify that an author is a U.S. voter.'),
        p(f'TF–IDF unigram/bigram features and NMF are fitted only to the early text. The objective gives the two media equal total weight; inverse document frequency still uses the full training sample. K=4, 6 and 8 are compared using training-word NPMI. A training-only diagnostic found that three repeated quotations formed a spurious cluster. Minimum document frequency was raised to five, and effective-support/concentration screens were added. K={summary["topic_k"]} passed: minimum effective support {winner.min_topic_effective_support:.1f}, maximum top-three-document mass {winner.max_topic_top_three_mass*100:.1f}%, seed cosine stability {winner.seed_stability_cosine:.3f}. These refinements are retrospective development, not preregistration.'),
        fig('topic_shares', 'NMF shares are mean normalized loadings, not calibrated topic probabilities or voter percentages. Labels summarize the top terms; clusters can mix policy and campaign language.', '.89'),
        table(['Pew voter issue', 'Pew importance', 'News rule mentions', 'HN rule mentions'], poll_rows, [.40,.20,.20,.20]),
        p('Pew gives registered-voter shares calling each issue very important (Sept. 28–Oct. 4; published Oct. 8). Mention columns use overlapping rules: democracy includes ethics, energy includes climate, and crime includes guns. They are broader than those survey items, so their percentages are not directly equivalent. The October poll is interpretation only. The pre-window July poll already put economic issues first (29% open-ended), then ethics/reform (9%), immigration (7%), healthcare (5%) and foreign policy (4%).'),
    ]))
    content.append(('A sentiment module with explicit measurement limits', [
        p('The primary score uses a pinned CardiffNLP Twitter RoBERTa model: probability of positive wording minus probability of negative wording. Headlines are scored as headlines; HN comments use their own text. Long records are split into token chunks and probabilities are averaged by token count, so no trailing text is discarded. This preserves one observation per record. The model’s 2018–2021 Twitter training domain differs from 2026 political headlines and HN.'),
        p('For topic k, each medium’s daily valence is the NMF-loading-weighted mean of document scores. A topic needs at least 0.5 document-equivalents of loading within that medium and period. The combined index is one half news plus one half social; a missing component leaves it missing. The broad index averages all model-covered text with the same source balance. No missing text is assigned neutral sentiment. NMF loadings express mixture strength, not independent topic probabilities.'),
        fig('sentiment_probability', 'Daily cutoffs are 16:00 New York. Source lines use trailing seven-day means for readability; all inference uses unsmoothed changes. Gray shading is the training window. Missing source days remain gaps.', '.98'),
        p('RSS dates do not establish historical ingestion time, so headlines become eligible at the following New York midnight. HN uses creation time. Text enters the first eligible calendar close, or the next actual trading close for equities; weekend text accumulates into Monday. The final cutoff excludes later Oct. 8 text and next-day RSS availability. Current text and adjusted-price vintages may differ from what was visible historically.'),
        p(audit_text),
    ]))
    result_rows = []
    target_names = {'polymarket_probability':'Polymarket Δquote', 'kalshi_probability':'Kalshi Δquote',
        'long_only_minus_spy':'Five-stock minus SPY', 'signed_beta_adjusted':'Signed beta-adjusted'}
    for _, row in broad.iterrows():
        result_rows.append([target_names[row.target], str(int(row.lag)), str(int(row.n)), f(row.pearson_r),
            f'[{f(row.block_ci_low)}, {f(row.block_ci_high)}]', f(row.holm_p,3)])
    topic_rows = []
    for _, topic in topics.iterrows():
        selected = core.loc[core['index'].eq('index_' + topic.topic) & core.target.eq('polymarket_probability') & core.lag.eq(0)].iloc[0]
        topic_rows.append([topic.display_label, str(int(selected.n)), f(selected.pearson_r), f(selected.holm_p,3)])
    content.append(('Association with sweep prices and market returns', [
        p('Polymarket’s YES asset directly prices Democratic House and Senate control. Kalshi’s DD combination is also joint, using the midpoint of closing hourly YES bid/ask quotes. The latest observation at or before 16:00 is used; observations older than 24 hours are missing. Polymarket resolves control following the November election, while Kalshi refers to February 2027. Settlement rules, spreads, liquidity and fees can all contribute to their difference. Quotes are market-implied prices, not a validated statistical forecast.'),
        p('Primary tests use changes in wording and changes in contract prices, or daily adjusted equity returns. Calendar days are used for contracts; actual cash sessions are used for stocks. Lag 0 is contemporaneous association; lag 1 pairs the prior period’s sentiment change with today’s outcome. Differences and lags are created before incomplete pairs are deleted. Levels are reported separately because common trends can generate misleading correlations.'),
        table(['Broad-index outcome', 'Lag', 'n', 'Pearson r', '95% block interval', 'Holm p'], result_rows, [.32,.06,.06,.12,.26,.18]),
        table(['NMF topic: same-day Polymarket change', 'n', 'r', 'Holm p'], topic_rows, [.58,.08,.14,.20]),
        p(f'OLS slope tests use a three-period Bartlett HAC covariance on the full time grid, including missing-period positions. Pointwise correlation intervals use 999 circular, five-period block resamples. Holm correction covers the fixed {summary["core_family_size"]}-test core family; insufficient-sample cases remain untested and count conservatively in that family. Intervals are not simultaneous, and a short, partly missing series limits both HAC and bootstrap reliability.'),
        fig('association_scatter', 'Assessment period only. Contract changes are percentage points; stock adjustment is the daily basket return less SPY. Scatterplots show the raw association without selecting a favorable trend line.', '.98'),
        p('Named policy issues are also measured with an explicit dictionary, separately from NMF: contextual scores on matching records, plus sentence-restricted VADER as a scope sensitivity. Their news-only and equal-medium contract/equity results form a separate exploratory family. Whole-record context can mix attitudes toward several issues. Sparse HN issue coverage often prevents daily topic inference. The notebook reports those sample sizes and tests, together with level correlations, source sensitivities, individual equities and direct quote-change/equity associations.'),
    ]))
    content.append(('Equity scenarios and market-adjusted performance', [
        table(['Equity', 'Sweep hypothesis', 'Policy channel and qualification'], [
            ['FSLR', 'Positive', 'Domestic solar manufacturing and Section 45X credits; trade policy can offset this channel.'],
            ['ENPH / RUN', 'Positive', 'Residential solar/storage demand and financing; a sweep cannot itself restore expired credits.'],
            ['GEO / CXW', 'Negative', 'Detention appropriations and contract growth; multiyear funding limits immediate reversal.'],
            ['UNH', 'Ambiguous', 'Public-program funding versus insurer regulation; diagnostic, outside the five-stock basket.'],
        ], [.16,.20,.64]),
        p('Signs follow company 10-Ks and Public Law 119-21, fixed before this project’s return inspection but after the study period. A congressional sweep faces presidential veto power, Senate procedure and existing funding. Residential Section 25D eligibility ended after 2025; CXW reports detention funding available through September 2029. Neither provision automatically reverses after the election.'),
        p('The long-only basket holds five stocks at 20% daily target weights. The directional basket is 50% gross long solar (one sixth each) and 50% gross short detention (minus one quarter each): 100% gross, zero net dollars. SPY subtraction cancels in this dollar-neutral basket; its additional adjustment uses constituent SPY betas frozen from 62 May–July sessions. Solar/TAN, detention/IWM and UNH/XLV differences provide diagnostics. IWM is a size proxy, not a detention-sector hedge.'),
        fig('basket_performance', 'Full Aug. 1–Oct. 8 window, using the July 31 prior close. Lower-panel values are sums of daily return differences, not compounded alpha. The signed strategy omits borrow, financing, fees and trading costs.', '.88'),
        table(['Equity', 'Assessment return', 'Sum: return − SPY', 'Sum: return − βSPY'],
            [[row.ticker, f'{row.total_return_pct:+.2f}%', f'{row.sum_spy_adjusted_pct:+.2f} pp', f'{row.sum_beta_adjusted_pct:+.2f} pp'] for _, row in perf.iterrows()], [.18,.25,.29,.28]),
        p(f'The table covers Aug. 24–Oct. 8 only; the chart covers the entire study. The broad-index same-day correlation is {f(basket.pearson_r)} with the five-stock SPY spread and {f(signed.pearson_r)} with the signed beta adjustment. Neither the raw path nor a positive sample correlation identifies election causality. Rates, energy prices, company earnings, contracts and sector news offer competing explanations.'),
    ]))
    content.append(('Conclusions, investment use and reproducibility', [
        p('I draw three conclusions. First, economic conditions and affordability are the clearest shared voter concern, with democracy and healthcare also prominent in the October importance ratings. Second, election-search topic frequencies are strongly shaped by campaigning and source selection, so they cannot replace survey evidence. Third, political wording, prediction-contract quotes and policy-exposed equity returns are different measurements. Their short-run association is too unstable and sample-dependent to treat as a causal or dependable trading signal.'),
        p(f'{inference}{diagnostic_text} Any nominally interesting topic, lag or individual equity remains exploratory. The model was selected without return information, but collection screens, topic diagnostics and the sentiment-method comparison were developed retrospectively. The assessment period therefore evaluates a frozen text model; it is not an untouched prospective trading test. The election outcome itself is still unobserved.'),
        p('My practical investment suggestion is a watchlist with explicit triggers: monitor affordability and healthcare discussion, both joint-control contracts, solar-credit legislation, detention appropriations and company guidance. FSLR/ENPH/RUN could benefit from supportive solar policy, while GEO/CXW could face slower expansion under tighter oversight. I would require a separately labeled human validation sample and forward testing before using this index to size positions. A hypothetical long-solar/short-detention scenario basket helps stress-test exposure, but the observed losses and omitted costs do not support recommending it as a profitable trade.'),
        p('The main limits are ranked and capped RSS discovery, a small technology-community social sample, unverified voter residence, near-duplicate paraphrases, uncertain historical text vintages and imperfect sentiment targets. Guardian access failed (401); Reddit and Bluesky failed (403). Broad GDELT requests were rate-limited (429); five records from a limited access probe are kept outside the study. No population-level voter sentiment, independent human accuracy, causal election effect or profitable out-of-sample strategy is claimed.'),
        p(f'The [GitHub repository]({REPO}) contains reusable acquisition, text modeling, sentiment, analysis and validation code; exact dependency pins; an executed notebook with aggregate outputs; and a truthful AI-use disclosure. The separate PDF is the report. Raw payloads, source texts, label audits, prices and derived datasets remain local and are excluded from Git. The snapshot manifest records local hashes. Fresh API downloads reproduce the method, while mutable feeds and prices may change the snapshot.'),
        p('Selected primary sources: [Pew October issue priorities]('+PEW+'); [Pew July open-ended priorities]('+JULY+'); [HN Algolia API](https://hn.algolia.com/api); [scikit-learn NMF](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html); [CardiffNLP model card](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest); [VADER authors’ implementation](https://github.com/cjhutto/vaderSentiment); [Polymarket event](https://polymarket.com/event/balance-of-power-2026-midterms); [Kalshi API](https://docs.kalshi.com/api-reference/market/get-market-candlesticks); [Public Law 119-21](https://www.govinfo.gov/content/pkg/PLAW-119publ21/html/PLAW-119publ21.htm).'),
        p('Company evidence: [First Solar 2025 annual report](https://s202.q4cdn.com/499595574/files/doc_financials/2025/ar/FSLR-2025-Annual-Report-web.pdf); [Enphase 2025 10-K](https://www.sec.gov/Archives/edgar/data/1463101/000146310126000013/enph-20251231.htm); [Sunrun 2025 10-K](https://investors.sunrun.com/filings-financials/sec-filings/content/0001628280-26-012289/run-20251231.htm); [GEO 2025 10-K](https://www.sec.gov/Archives/edgar/data/923796/000119312526071747/geo-20251231.htm); [CoreCivic 2025 10-K](https://ir.corecivic.com/static-files/3f95f305-eeb0-4aba-abbd-d1db10deca6a); [UnitedHealth 2025 10-K](https://www.unitedhealthgroup.com/content/dam/UHG/PDF/investors/2025/UNH-Q4-2025-Form-10-K.pdf).'),
    ]))
    return content
