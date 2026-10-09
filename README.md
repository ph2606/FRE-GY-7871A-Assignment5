# FRE-GY 7871A — Assignment 5

Panagiotis Housos | ph2606 | NLP and the Investment Process | Fall 2026

[Open the executed notebook](assignment5.ipynb). The separate six-page PDF,
`Panagiotis_Housos_Assignment5.pdf`, studies **August 1–October 8, 2026**, with
information ending at **16:00 America/New_York on October 8**. Model training
uses text available by August 21; later dates assess the frozen text model.
The election outcome is still unobserved. This is a retrospective research
exercise, not a prospective trading backtest.

## Submitted findings

The final information cutoff leaves 3,232 texts: 2,953 headlines and 279 social
records. The four-topic NMF model emphasizes campaign discourse; Pew's survey
benchmark identifies the economy, democracy and healthcare as leading voter
priorities. Contextual sentiment agreed with 41/52 eligible AI-reference labels,
versus VADER's 22/52, with limited positive-class support and no human-accuracy
claim.

Broad sentiment-change correlations are −0.18 with Polymarket quote changes,
+0.10 with Kalshi changes (44 paired days), and −0.05 with the five-stock basket
minus SPY (33 sessions). One of 40 core tests survives Holm correction: same-day
Trump/tax/campaign-topic wording versus the signed beta-adjusted basket,
`r=0.36`, `n=28`, adjusted `p=0.023`. It is contemporaneous, and uncertainty
depends on covariance assumptions; it is insufficient evidence for a timing
strategy. No policy-issue or individual-equity exploratory test survives its
own family correction.

Full-window sweep quotes rose from 44.5% to 63.5% on Polymarket and 45.5% to
61.5% on Kalshi. The five-stock basket returned −7.44%, the signed scenario
basket −11.27%, and SPY +3.86%. These observations do not identify an election's
causal effect. The investment conclusion is a policy watchlist and forward
validation, rather than a sentiment-driven position from this sample.

## Data and topic modeling

Google News RSS supplies dated headlines and HN Algolia supplies Hacker News
comments and stories. Search is broad election discovery, with no return or
sentiment words in the queries. Country, election context, language, timestamps
and exact normalized duplicates are screened. HN comments use their own text;
thread titles can establish relevance but never enter comment sentiment.
Ranked RSS requests hit caps, so the news sample is not exhaustive. HN is a
technology-community convenience sample, with no verified voter identity.
Guardian, Reddit and Bluesky probes failed. Broader GDELT calls were rate-limited;
five records from a small successful probe remain outside the primary analysis.

Raw response bytes, SHA256 digests, retrieval clocks, provider dates, query caps
and row-level exclusions are retained locally. RSS headlines become eligible at
the following New York midnight, a conservative retrospective availability
proxy; HN uses creation timestamps. Historical ingestion clocks and original
unedited wording are unavailable. Text after the final information cutoff is
excluded before topic summaries and correlations.

TF–IDF and nonnegative matrix factorization use only the training text. Media
receive equal total NMF objective weight, while IDF uses all training documents.
K=4, 6 and 8 are compared on training-word NPMI with effective-support and
concentration screens. A training-only inspection found repeated quotations
forming a spurious topic, motivating minimum document frequency five and those
screens. These are disclosed retrospective method refinements. NMF loadings are
normalized mixture weights, not calibrated topic probabilities.

Pew's July and October surveys distinguish reported voter priorities from
sample discourse. October polling is interpretation only and never supplies
historical signal weights. A separate overlapping policy dictionary measures
economy, democracy/ethics, healthcare, immigration, foreign policy and additional
issues that unsupervised campaign clusters may miss. Dictionary mention shares
and survey importance percentages answer different questions.

## Sentiment and inference

The primary wording measure is the pinned
[`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
model at revision `3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7`, using
`P(positive) − P(negative)`. Its CC-BY-4.0 model card and every downloaded file's
hash are cached. Weights load as tensors with no remote code. All text tokens
are retained in nonoverlapping 510-token chunks; chunk probabilities are
token-weighted before producing one score per document. Cross-chunk context is
unavailable, and probabilities are not calibrated for 2026 political text.
VADER is a lexical sensitivity, including sentence-restricted policy wording.

A frozen 60-text AI-reference diagnostic checks errors. Eight labels fall outside
the final eligible corpus after timing/relevance screens, so model comparison uses the same 52 eligible
records. Labels are not independent human ground truth; ambiguity, sarcasm,
quotation and sentiment targets limit interpretation. Model comparison was
consulted during development, so it is not an untouched accuracy test. No
party-support or voter-intention accuracy is claimed. The executed notebook
preserves aggregate diagnostic results; detailed labels remain local.

Each topic index combines its within-news and within-social loading-weighted
valence equally. Missing components remain missing. Contract observations use
daily 16:00 New York cutoffs, while equities use actual trading closes and roll
weekend text into the next session. Lags and differences are formed on the full
grid before incomplete pairs are dropped. The primary regressions compare
changes, with levels shown separately. Bartlett HAC uses three actual grid
periods, including missing positions; correlation intervals use 999 circular
five-period block resamples. Holm correction covers the complete fixed core
family, counting untestable cases conservatively. Policy issues and individual
equities have separately disclosed exploratory families.

## Contracts and equities

Both contracts directly refer to Democratic control of the **House and Senate**:
Polymarket event `balance-of-power-2026-midterms`, market `562828`, and Kalshi
`KXBALANCEPOWERCOMBO-27FEB-DD`. Marginal chamber prices are never multiplied.
Polymarket history uses the YES asset; Kalshi uses hourly closing YES bid/ask
midpoints. Only observations at or before the cutoff, at most 24 hours old, are
eligible. Kalshi's February 2027 control terms differ from Polymarket's
post-election resolution. Quote prices need not be calibrated probabilities.

The long-only basket targets 20% each in FSLR, ENPH, RUN, GEO and CXW. The
directional scenario basket is 50% gross long solar (one sixth each) and 50%
gross short detention (minus one quarter each). Solar beneficiaries and
detention headwinds are conditional policy hypotheses grounded in company
filings and Public Law 119-21. A congressional sweep does not itself restore
expired residential credits, eliminate multiyear detention funding or remove
presidential veto power. UNH is a diagnostic because its policy channels have
ambiguous signs.

Yahoo adjusted daily prices use July 31 as the prior close. SPY is the broad
benchmark; TAN, IWM and XLV supply diagnostics. All expected NYSE sessions are
restored before returns are calculated, and missing constituents never silently
reweight baskets. Betas use only 62 May–July sessions. Simple market adjustment
is `r_i − r_SPY`; beta adjustment is `r_i − beta_i*r_SPY`. SPY subtraction
cancels in the zero-net-dollar signed basket; the frozen-beta adjustment
addresses unequal market exposure. Return differences and their sums are not
CAPM alpha. Financing, short borrow, fees and trading costs are omitted.

## Reproduce

Use Python 3.12. From the repository root in PowerShell:

```powershell
python -m venv .venv
$assignmentPython = '.\.venv\Scripts\python.exe'
& $assignmentPython -m pip install -r requirements-lock.txt
& $assignmentPython -m ipykernel install --user --name assignment5 --display-name 'Python 3 (Assignment 5)'

& $assignmentPython scripts/01_collect_text.py
& $assignmentPython scripts/02_collect_markets.py

# Initial text model establishes the eligible IDs/text for contextual scoring.
& $assignmentPython -m src.text_model --lexical-only
& $assignmentPython scripts/03b_contextual_score.py
& $assignmentPython scripts/03_analyze.py
& $assignmentPython -m pytest tests -q

# Optional local diagnostic: requires separately frozen blind labels.
# & $assignmentPython -m src.sentiment_review

& $assignmentPython scripts/04_build_deliverables.py
& $assignmentPython scripts/05_execute_notebook.py
xelatex -interaction=nonstopmode -halt-on-error report.tex
xelatex -interaction=nonstopmode -halt-on-error report.tex
Copy-Item -LiteralPath report.pdf -Destination Panagiotis_Housos_Assignment5.pdf
& $assignmentPython scripts/06_freeze_and_validate.py
```

The contextual weights require about 500 MB on first download. CPU inference is
supported; the package lock specifies the official CPU PyTorch wheel source.
Cached requests and model files are hash-verified. An intentional market refresh
supports `--refresh` and preserves old raw vintages; text acquisition reuses its
frozen request cache. Newly acquired mutable feeds, wording and adjusted prices
may change results. Fresh downloads reproduce the method, not necessarily the
exact submitted snapshot. `SNAPSHOT.md` binds the submission to local input
hashes. Rebuild report and notebook after changing inputs or measurement.

XeLaTeX uses Cambria/Calibri with Latin Modern fallbacks. The shared narrative in
`scripts/study_content.py` supplies the six-page report and notebook; exhibits
come from computed outputs. Detailed source URLs are linked there and in the
notebook. Diagnostic labels are deliberately not distributed; fresh validation
requires fresh blind labels, while the submitted notebook retains aggregate
evidence. Synthetic fixtures only test software and never enter the study.

## Submission and repository policy

Submit the separate PDF with this repository URL. Git retains source, tests,
dependencies, the executed notebook with aggregate exhibits, `SNAPSHOT.md` and
[AI_USE.md](AI_USE.md). **No datasets are committed**: raw documents, social
records, prices, detailed qualitative labels, derived tables, model weights,
caches and generated report files remain ignored. No trade or institutional
submission was made.
