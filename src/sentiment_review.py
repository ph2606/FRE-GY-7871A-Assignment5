"""Recompute an explicitly AI-reference diagnostic from frozen local labels.

Reference labels are empirical annotation data and remain outside Git. This
module never manufactures labels or treats AI agreement as human accuracy.
"""
import json
import pandas as pd
from sklearn.metrics import f1_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from .config import DATA, OUT


def audit_reviews():
    label_path = DATA / 'sentiment_review_labels.csv'
    if not label_path.exists():
        raise FileNotFoundError(
            'Frozen local AI-review labels are required; no annotation data is '
            'committed. A fresh clone must collect and label its own blind sample.'
        )
    labels = pd.read_csv(label_path)
    if labels.id.duplicated().any():
        raise ValueError('Review ids must be unique.')
    classes = ['negative', 'neutral', 'positive']
    if not labels.ai_reference_class.isin(classes).all():
        raise ValueError('Review labels must be negative, neutral or positive.')
    if not labels.scores_blinded_during_labeling.eq(True).all():
        raise ValueError('Diagnostic requires labels frozen before score comparison.')
    score_path = DATA / 'scored_texts.csv'
    columns = pd.read_csv(score_path, nrows=0).columns
    # Contextual scoring can replace the primary sentiment fields. Preserve
    # this originally blinded diagnostic as a VADER baseline comparison.
    value_field = 'vader' if 'vader' in columns else 'sentiment'
    class_field = 'vader_class' if 'vader_class' in columns else 'sentiment_class'
    scores = pd.read_csv(score_path, usecols=['id', value_field, class_field])
    scores = scores.rename(columns={value_field: 'sentiment',
                                   class_field: 'sentiment_class'})
    audit = labels.merge(scores, on='id', how='left', validate='one_to_one')
    audit['retained_in_final_corpus'] = audit.sentiment_class.notna()
    # A later timestamp screen can exclude a previously reviewed record. Keep
    # all frozen labels, recompute its unchanged lexical baseline from own text,
    # and report the final-corpus subset separately for a fair model comparison.
    missing = ~audit.retained_in_final_corpus
    analyzer = SentimentIntensityAnalyzer()
    for position, row in audit.loc[missing].iterrows():
        value = analyzer.polarity_scores(row.analysis_text)['compound']
        audit.loc[position, 'sentiment'] = value
        audit.loc[position, 'sentiment_class'] = (
            'negative' if value <= -.05 else 'positive' if value >= .05 else 'neutral')
    audit['agreement'] = audit.ai_reference_class.eq(audit.sentiment_class)
    audit.to_csv(DATA / 'sentiment_review_comparison.csv', index=False)
    confusion = pd.crosstab(audit.ai_reference_class, audit.sentiment_class)
    confusion = confusion.reindex(index=classes, columns=classes, fill_value=0)
    confusion.to_csv(OUT / 'sentiment_review_confusion.csv')
    def grouped(field):
        return audit.groupby(field).agreement.agg(['count', 'sum', 'mean']).rename(
            columns={'count': 'n', 'sum': 'agreements', 'mean': 'agreement'}
        ).to_dict('index')
    unambiguous = audit.loc[~audit.ambiguous_valence, 'agreement']
    retained = audit.loc[audit.retained_in_final_corpus]
    summary = {
        'classifier': 'VADER 3.3.2 lexical/rule-based baseline',
        'n': len(audit), 'agreements': int(audit.agreement.sum()),
        'ai_reference_agreement': float(audit.agreement.mean()),
        'ai_reference_macro_f1': float(f1_score(
            audit.ai_reference_class, audit.sentiment_class, labels=classes,
            average='macro', zero_division=0)),
        'ambiguous_n': int(audit.ambiguous_valence.sum()),
        'party_target_mismatch_n': int(audit.party_target_mismatch.sum()),
        'unambiguous_n': len(unambiguous),
        'unambiguous_agreement': float(unambiguous.mean()) if len(unambiguous) else None,
        'by_source': grouped('source_type'), 'by_split': grouped('review_split'),
        'retained_final_corpus_n': len(retained),
        'excluded_before_final_scoring_n': int(missing.sum()),
        'retained_final_corpus_agreements': int(retained.agreement.sum()),
        'retained_final_corpus_agreement': float(retained.agreement.mean())
            if len(retained) else None,
        'focus_counts': audit.midterm_focus.value_counts().to_dict(),
        'interpretation': (
            'AI-reference agreement diagnostic only, not human-validated accuracy. '
            'Labels were frozen blind to VADER scores, topics and market outcomes. '
            'Stratification and subjective reference labels prevent interpreting '
            'this as corpus or voter-population accuracy. The original diagnostic '
            'retains all frozen cases; final-corpus subset agreement is reported '
            'separately when subsequent timestamp filtering excludes cases.'
        ),
    }
    (OUT / 'sentiment_review_summary.json').write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    return summary


if __name__ == '__main__':
    print(json.dumps(audit_reviews(), indent=2))
