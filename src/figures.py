"""Publication figures built only from computed observations."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from .config import DATA, OUT, START, TRAIN_END, END

NAVY = '#17334D'
TEAL = '#168C8C'
GOLD = '#D59138'
RED = '#B65B59'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelcolor': NAVY, 'text.color': NAVY, 'axes.edgecolor': '#BBC5CD',
    'grid.color': '#E8ECEF', 'axes.titleweight': 'bold', 'savefig.facecolor': 'white'})


def save(figure, name):
    figure.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    figure.savefig(OUT / f'{name}.png', bbox_inches='tight', dpi=170)
    plt.close(figure)


def decorate_time(ax):
    ax.axvspan(pd.Timestamp(START), pd.Timestamp(TRAIN_END), color='#E8ECEF', alpha=.8, zorder=-2)
    ax.axvline(pd.Timestamp(TRAIN_END), color='#83929C', lw=.8, ls='--')
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.grid(axis='y', lw=.5)


def build():
    calendar = pd.read_csv(OUT / 'calendar_panel.csv', index_col='date', parse_dates=True)
    market = pd.read_csv(OUT / 'market_panel.csv', index_col='date', parse_dates=True)
    topics = pd.read_csv(OUT / 'topic_summary.csv')
    display_labels = topics.get('display_label', topics.label)
    y = np.arange(len(topics))
    fig, ax = plt.subplots(figsize=(7.6, 3.3))
    ax.barh(y + .17, topics.news_share * 100, height=.30, color=NAVY, label='News headlines')
    ax.barh(y - .17, topics.social_share * 100, height=.30, color=TEAL, label='Hacker News text')
    ax.set_yticks(y, display_labels)
    ax.invert_yaxis()
    ax.set_xlabel('Mean normalized topic loading (%)')
    ax.legend(frameon=False, ncol=2, loc='lower left', bbox_to_anchor=(0, 1.01), borderaxespad=0)
    ax.grid(axis='x', lw=.5)
    fig.tight_layout()
    save(fig, 'topic_shares')

    fig, axes = plt.subplots(3, 1, figsize=(7.6, 4.7), sharex=True, gridspec_kw={'height_ratios': [1.2, 1.1, .7]})
    for source, color, label in [('news', NAVY, 'News'), ('social', TEAL, 'Hacker News')]:
        smooth = calendar[f'{source}_all'].rolling(7, min_periods=4).mean().where(calendar[f'{source}_all'].notna())
        axes[0].plot(calendar.index, smooth, lw=1.4, color=color, label=label)
    axes[0].plot(calendar.index, calendar.index_all, color='#AAB4BC', lw=.6, alpha=.6, label='Daily combined')
    axes[0].axhline(0, color='#83929C', lw=.7)
    axes[0].set_ylabel('Wording valence')
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    axes[1].plot(calendar.index, calendar.polymarket_probability * 100, color=TEAL, label='Polymarket')
    axes[1].plot(calendar.index, calendar.kalshi_probability * 100, color=GOLD, label='Kalshi')
    axes[1].set_ylabel('Joint DD quote (%)')
    axes[1].legend(frameon=False, ncol=2, fontsize=8)
    axes[2].bar(calendar.index, calendar.news_n.fillna(0), color=NAVY, width=.8, label='News')
    axes[2].bar(calendar.index, calendar.social_n.fillna(0), bottom=calendar.news_n.fillna(0), color=TEAL, width=.8, label='HN')
    axes[2].set_ylabel('Items')
    for ax in axes:
        decorate_time(ax)
    fig.tight_layout(h_pad=.5)
    save(fig, 'sentiment_probability')

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.5))
    x = calendar.index_all.diff().loc[calendar.index > TRAIN_END]
    y = (calendar.polymarket_probability.diff() * 100).reindex(x.index)
    valid = x.notna() & y.notna()
    axes[0].scatter(x.loc[valid], y.loc[valid], s=16, alpha=.65, color=TEAL)
    axes[0].set_ylabel('Change in Polymarket quote (pp)')
    x2 = market.index_all.diff().loc[market.index > TRAIN_END]
    y2 = (market.long_only_minus_spy * 100).reindex(x2.index)
    valid = x2.notna() & y2.notna()
    axes[1].scatter(x2.loc[valid], y2.loc[valid], s=18, alpha=.7, color=NAVY)
    axes[1].set_ylabel('Five-stock return minus SPY (pp)')
    for ax in axes:
        ax.set_xlabel('Change in combined valence')
        ax.axhline(0, color='#BBC5CD', lw=.7)
        ax.axvline(0, color='#BBC5CD', lw=.7)
        ax.grid(lw=.4)
    fig.tight_layout(w_pad=2)
    save(fig, 'association_scatter')

    fig, axes = plt.subplots(2, 1, figsize=(7.6, 3.8), sharex=True)
    for field, color, label in [('long_only_return_pct', NAVY, 'Five-stock basket'), ('spy_return_pct', '#83929C', 'SPY'), ('signed_return_pct', TEAL, 'Solar long / detention short')]:
        wealth = ((1 + market[field] / 100).cumprod() - 1) * 100
        axes[0].plot(market.index, wealth, label=label, color=color, lw=1.4)
    axes[0].set_ylabel('Cumulative return (%)')
    axes[0].legend(frameon=False, fontsize=7.8, ncol=3)
    axes[1].plot(market.index, market.long_only_minus_spy.cumsum() * 100, color=NAVY, label='Five-stock minus SPY')
    axes[1].plot(market.index, market.signed_beta_adjusted.cumsum() * 100, color=TEAL, label='Signed basket minus frozen beta × SPY')
    axes[1].set_ylabel('Sum of adjusted returns (pp)')
    axes[1].axhline(0, color='#83929C', lw=.7)
    axes[1].legend(frameon=False, fontsize=7.8, ncol=2)
    for ax in axes:
        decorate_time(ax)
    fig.tight_layout(h_pad=.5)
    save(fig, 'basket_performance')


if __name__ == '__main__':
    build()
