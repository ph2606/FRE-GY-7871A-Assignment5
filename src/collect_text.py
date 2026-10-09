"""Acquire a dated, auditable midterm headline and Hacker News text sample.

Queries contain election context, never sentiment or basket returns. Publication
times are provider claims; historical immutable text vintages are unavailable.
RSS records become eligible only at the following New York midnight, while HN
records use their own creation time. This is retrospective association data.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlencode, urlsplit
from zoneinfo import ZoneInfo
import argparse
import hashlib
import html
import json
import re
import time
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
from langdetect import DetectorFactory, detect_langs
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/text'
NY = ZoneInfo('America/New_York')
OFFLINE = False
NEWS_QUERIES = {
    'midterms': '(midterm OR "congressional election") (US OR "United States" OR Democrat OR Republican OR Trump)',
    'congress': '(House OR Senate) (election OR voters) (Democrats OR Republicans)',
}
HN_QUERIES = ['midterm', 'congressional election', 'Senate election', 'House election', 'election', 'voter', 'ballot']
ELECTION = re.compile(r'\b(?:mid[- ]?terms?|congressional elections?|elections?|voters?|ballot|voting|campaign|polls?|primar(?:y|ies))\b', re.I)
US = re.compile(r'\b(?:(?-i:U\.?S\.?)|United States|America(?:n)?|Trump|Vance|Democrats?|Republicans?|GOP|MAGA|Congress|Senate|senators?|House of Representatives|DNC|RNC|ICE|Harris|Biden)\b', re.I)
FOREIGN = re.compile(r'\b(?:India|Indian|Pakistan|Bangladesh|Nepal|Philippines|Philippine|Canada|Canadian|Britain|British|UK|Knesset|French|France|Brazil|Argentina|Australia)\b', re.I)
CONGRESSIONAL_CONTEXT = re.compile(r'\b(?:midterms?|mid[- ]term elections?|congressional elections?)\b|'
    r'\b(?:House|Senate|Congress)\b.{0,55}\b(?:elections?|control|majorit(?:y|ies)|seats|ballot)\b|'
    r'\b(?:elections?|control|majorit(?:y|ies)|seats|ballot)\b.{0,55}\b(?:House|Senate|Congress)\b', re.I)
DATED_ELECTION_CONTEXT = re.compile(
    r'\b(?:2026|November)\b[^\n.!?]{0,80}\b(?:elections?|vote|voters?|voting|ballots?|polls?|primar(?:y|ies))\b|'
    r'\b(?:elections?|vote|voters?|voting|ballots?|polls?|primar(?:y|ies))\b[^\n.!?]{0,80}\b(?:2026|November)\b', re.I)
NONPOLITICAL_ELECTION = re.compile(
    r'\bNLRB\b|\b(?:union|workplace|board of directors)\b[^\n.!?]{0,40}\b(?:elections?|ballots?)\b|'
    r'\b(?:elections?|votes?|voting)\b[^\n.!?]{0,40}\b(?:union|board of directors)\b', re.I)
OTHER_ELECTION_YEAR = re.compile(r'\b(?:19\d{2}|20(?:0\d|1\d|2[0-589]|[3-9]\d))\s+(?:(?:US|U\.S\.|presidential|midterm)\s+)?elections?\b', re.I)
WORDS = re.compile(r'[A-Za-z]+')
DetectorFactory.seed = 787105


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def clean(value):
    value = value or ''
    if '<' in value:
        value = BeautifulSoup(value, 'html.parser').get_text(' ', strip=True)
    return re.sub(r'\s+', ' ', html.unescape(value)).strip()


def fetch(url, name, params=None):
    """Cache original bytes and a verifiable retrieval ledger, including errors."""
    full = url + ('?' + urlencode(params) if params else '')
    key = hashlib.sha256(full.encode()).hexdigest()[:24]
    path = RAW / name / (key + '.payload')
    meta_path = path.with_suffix('.metadata.json')
    if path.exists() and meta_path.exists():
        payload, meta = path.read_bytes(), json.loads(meta_path.read_text(encoding='utf-8'))
        if hashlib.sha256(payload).hexdigest() != meta['sha256']:
            raise ValueError(f'Raw payload digest mismatch: {path}')
        return payload, meta
    if OFFLINE:
        raise FileNotFoundError(f'Offline collection requires both cached payload and metadata: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(url=full, retrieved_at=utcnow(), status='failed', errors=[])
    payload = b''
    for attempt in range(2):
        try:
            response = requests.get(full, timeout=40, headers={'User-Agent': 'NLP-academic-assignment/1.0 (personal research)'})
            payload = response.content
            meta.update(status=response.status_code, retrieved_at=utcnow(), final_url=response.url)
            if response.status_code in (429, 500, 502, 503, 504) and attempt == 0:
                # Every failed payload is retained, not overwritten by the retry.
                failed = path.with_name(path.stem + '.attempt1.payload')
                failed.write_bytes(payload)
                failed.with_suffix('.metadata.json').write_text(json.dumps(dict(meta, sha256=hashlib.sha256(payload).hexdigest()), indent=2), encoding='utf-8')
                time.sleep(3)
                continue
            break
        except requests.RequestException as exc:
            meta['errors'].append(str(exc))
            if attempt == 0:
                time.sleep(1)
    path.write_bytes(payload)
    meta.update(sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload), raw_path=path.relative_to(ROOT).as_posix())
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return payload, meta


def rss_job(day, family):
    following = day + timedelta(days=1)
    query = f'{NEWS_QUERIES[family]} after:{day:%Y-%m-%d} before:{following:%Y-%m-%d}'
    payload, meta = fetch('https://news.google.com/rss/search', 'google_rss', {'q': query, 'hl': 'en-US', 'gl': 'US', 'ceid': 'US:en'})
    rows = []
    audit = dict(meta, provider='Google News RSS', query=query, query_date=day.date().isoformat(), family=family, returned=0, capped=False)
    if meta['status'] == 200:
        try:
            items = ET.fromstring(payload).findall('./channel/item')
            audit.update(returned=len(items), capped=len(items) >= 100)
            for item in items:
                title = clean(item.findtext('title'))
                source = clean(item.findtext('source'))
                if source and title.endswith(' - ' + source):
                    title = title[:-len(source)-3]
                link = item.findtext('link') or ''
                try:
                    stamp = parsedate_to_datetime(item.findtext('pubDate')).astimezone(timezone.utc)
                except (TypeError, ValueError):
                    stamp = None
                available = (datetime.combine(stamp.astimezone(NY).date() + timedelta(days=1), datetime.min.time(), tzinfo=NY).astimezone(timezone.utc).isoformat() if stamp else '')
                rows.append(dict(id='gn_' + hashlib.sha256(link.encode()).hexdigest()[:20], source=source,
                    source_type='news', provider='Google News RSS', published_at=stamp.isoformat() if stamp else '',
                    available_at=available, available_at_basis='next New York midnight after publisher publication; conservative retrospective proxy',
                    text=title, title=title, url=link, author='', thread='', record_kind='headline', text_scope='headline',
                    query=family, query_date=day.date().isoformat(), language_basis='English US RSS edition plus seeded langdetect screen',
                    raw_path=meta['raw_path'], sha256=meta['sha256'], retrieved_at=meta['retrieved_at']))
        except ET.ParseError as exc:
            audit.update(parse_error=str(exc))
    return rows, [audit]


def hn_shard(query, left, right):
    params = {'query': query, 'numericFilters': f'created_at_i>={left},created_at_i<{right}', 'hitsPerPage': 1000,
              'page': 0, 'typoTolerance': 'false', 'queryType': 'prefixNone'}
    payload, meta = fetch('https://hn.algolia.com/api/v1/search_by_date', 'hn_algolia', params)
    audit = dict(meta, provider='HN Algolia', query=query, start_epoch=left, end_epoch=right,
                 returned=0, reported_hits=None, capped=False, used_as_leaf=True)
    if meta['status'] != 200:
        return [], [audit]
    try:
        obj = json.loads(payload)
        hits = obj.get('hits', [])
        audit.update(returned=len(hits), reported_hits=obj.get('nbHits'), exhaustive=obj.get('exhaustiveNbHits'))
        if obj.get('nbHits', len(hits)) > len(hits) or len(hits) >= 1000:
            audit.update(capped=True, used_as_leaf=False)
            if right-left <= 1:
                raise RuntimeError('Algolia cap unresolved within a one-second shard')
            middle = (left+right)//2
            a, aa = hn_shard(query, left, middle)
            b, ba = hn_shard(query, middle, right)
            return a+b, [audit]+aa+ba
        rows = []
        for hit in hits:
            kind = 'comment' if 'comment' in hit.get('_tags', []) else 'story'
            title = clean(hit.get('title')) if kind == 'story' else ''
            body = clean(hit.get('comment_text') if kind == 'comment' else hit.get('story_text'))
            text = ' '.join(x for x in (title, body) if x)
            ident = str(hit['objectID'])
            rows.append(dict(id='hn_'+ident, source='Hacker News', source_type='social', provider='HN Algolia',
                published_at=hit.get('created_at', ''), available_at=hit.get('created_at', ''),
                available_at_basis='own HN record creation time; body as retrieved, original unedited vintage unavailable',
                text=text, title=title, url='https://news.ycombinator.com/item?id='+ident, author=hit.get('author') or '',
                thread=str(hit.get('story_id') or (ident if kind == 'story' else '')), story_title=clean(hit.get('story_title')),
                record_kind=kind, text_scope='own HN comment only' if kind == 'comment' else 'own HN story title and text',
                query=query, query_date='', language_basis='seeded langdetect screen of own text',
                raw_path=meta['raw_path'], sha256=meta['sha256'], retrieved_at=meta['retrieved_at']))
        return rows, [audit]
    except (ValueError, KeyError) as exc:
        audit['parse_error'] = str(exc)
        return [], [audit]


def gdelt_job(day):
    """Independent news-API access audit; titles kept in a separate supplement.

    GDELT seendate is first indexing time, never mislabeled publication time.
    Its capped, uneven supplement is excluded from the primary text corpus.
    """
    end = day + timedelta(days=1)
    params = dict(query='(midterm OR "congressional election") sourcelang:english', mode='artlist', format='json',
                  startdatetime=day.strftime('%Y%m%d%H%M%S'), enddatetime=end.strftime('%Y%m%d%H%M%S'), maxrecords=250, sort='DateAsc')
    payload, meta = fetch('https://api.gdeltproject.org/api/v2/doc/doc', 'gdelt', params)
    audit = dict(meta, provider='GDELT DOC', query=params['query'], query_date=day.date().isoformat(), returned=0, capped=False)
    rows = []
    if meta['status'] == 200:
        try:
            obj = json.loads(payload)
            articles = obj.get('articles', [])
            audit.update(returned=len(articles), capped=len(articles)>=250)
            for item in articles:
                rows.append(dict(item, raw_path=meta['raw_path'], sha256=meta['sha256'], retrieved_at=meta['retrieved_at']))
        except ValueError:
            audit['parse_error'] = payload.decode(errors='replace')[:300]
    return rows, [audit]


def access_probes(start, end):
    probes = {
        'guardian': ('https://content.guardianapis.com/search', {'q':'midterm', 'from-date':str(start.date()), 'to-date':str((end-timedelta(days=1)).date()), 'api-key':'test', 'page-size':1}),
        'reddit': ('https://www.reddit.com/search.json', {'q':'midterm', 'sort':'new', 'limit':2}),
        'bluesky': ('https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts', {'q':'midterm', 'since':start.astimezone(timezone.utc).isoformat(), 'until':end.astimezone(timezone.utc).isoformat(), 'limit':2, 'sort':'latest'}),
    }
    out = []
    for name, (url, params) in probes.items():
        _, meta = fetch(url, 'access_probes/'+name, params)
        out.append(dict(meta, name=name, usable_corpus=False, reason='probe only; credentials not available or access denied'))
    return out


def polling_benchmark():
    """Save primary-source survey snapshots; used only to validate interpretation."""
    links = {
        'pew_oct8_issues': 'https://www.pewresearch.org/politics/2026/10/08/economy-ranks-as-top-issue-in-midterms-for-all-voters-but-partisans-differ-over-the-importance-of-other-issues/',
        'pew_oct8_methodology': 'https://www.pewresearch.org/politics/2026/10/08/midterms-2026-methodology/',
        'pew_jul23_priorities': 'https://www.pewresearch.org/politics/2026/07/23/as-the-2026-midterms-approach-economy-is-front-and-center/',
        'pew_oct8_topline': 'https://www.pewresearch.org/wp-content/uploads/sites/20/2026/10/PP_2026.10.8_midterms-economy_topline.pdf',
    }
    audit, benchmark = [], []
    for name, url in links.items():
        payload, meta = fetch(url, 'polling/'+name)
        audit.append(dict(name=name, **meta))
        if name == 'pew_oct8_issues' and meta['status'] == 200:
            tables = BeautifulSoup(payload, 'html.parser').select('table')
            if not tables:
                raise ValueError('Pew voter priority table unavailable; inspect frozen primary payload')
            for row in tables[0].select('tr'):
                values = [clean(cell.get_text(' ', strip=True)) for cell in row.select('td')]
                if len(values) < 2 or not values[1].isdigit():
                    continue
                benchmark.append(dict(issue=values[0], percent_very_important=int(values[1]), population='registered US voters',
                    published_date='2026-10-08', field_start='2026-09-28', field_end='2026-10-04', url=url,
                    raw_path=meta['raw_path'], sha256=meta['sha256']))
    (ROOT/'outputs/polling_source_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    if benchmark:
        pd.DataFrame(benchmark).to_csv(ROOT/'data/processed/pew_voter_priorities.csv', index=False)


def small_gdelt_probe():
    """One separately cached, limited API sample after bulk access fails."""
    params = dict(query='midterm sourcelang:english', mode='artlist', format='json',
                  startdatetime='20261001000000', enddatetime='20261002000000', maxrecords=5)
    payload, meta = fetch('https://api.gdeltproject.org/api/v2/doc/doc', 'gdelt_access_check', params)
    audit = dict(meta, provider='GDELT DOC', query=params['query'], query_date='2026-10-01',
                 returned=0, capped=False, scope='limited serial access probe only')
    (ROOT/'outputs/gdelt_serial_access_probe.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    try:
        found = json.loads(payload).get('articles', []) if meta['status'] == 200 else []
        audit.update(returned=len(found), capped=len(found)>=5)
        return [dict(x, raw_path=meta['raw_path'], sha256=meta['sha256'], retrieved_at=meta['retrieved_at']) for x in found], audit
    except ValueError:
        return [], audit


def screen(row, start, end):
    text = row['text']
    inherited_title = row.get('story_title', '')
    context = text + ' ' + (inherited_title if isinstance(inherited_title, str) else '')
    # A year or country in a citation URL is not the subject of the comment.
    context = re.sub(r'https?://\S+', ' ', context, flags=re.I)
    try:
        stamp = pd.Timestamp(row['published_at'])
        if stamp.tzinfo is None or not start <= stamp.to_pydatetime() < end:
            return 'outside requested publication window or missing timezone'
    except (ValueError, TypeError):
        return 'invalid publication timestamp'
    if len(WORDS.findall(text)) < (8 if row['source_type'] == 'social' else 4):
        return 'too little own text'
    if not ELECTION.search(context):
        return 'no explicit election context'
    # Midterm + US context excludes school midterms and foreign elections.
    if not US.search(context):
        return 'no explicit US political context'
    if row['source_type'] == 'social':
        explicit_congress = bool(CONGRESSIONAL_CONTEXT.search(context))
        if NONPOLITICAL_ELECTION.search(context) and not explicit_congress:
            return 'nonpolitical election without explicit congressional-midterm context'
        if not explicit_congress and not DATED_ELECTION_CONTEXT.search(context):
            return 'no explicit dated-election or congressional-midterm context'
    if OTHER_ELECTION_YEAR.search(context) and not re.search(r'\b(?:2026|midterms?|upcoming|this November)\b', context, re.I):
        return 'other election year without current-midterm context'
    if row['source_type'] == 'news' and FOREIGN.search(context) and not re.search(r'\b(?:Trump|United States|US|U\.S\.|Democrats?|Republicans?)\b', context, re.I):
        return 'foreign election ambiguity'
    try:
        languages = detect_langs(text)
        row['language_detected'] = languages[0].lang
        row['language_confidence'] = languages[0].prob
        if languages[0].lang != 'en':
            return 'language detector not English'
    except Exception:
        return 'language detector failed'
    return ''


def collect(start='2026-08-01', end='2026-10-08', workers=6, gdelt=True, offline=False):
    global OFFLINE
    OFFLINE = offline
    left = datetime.fromisoformat(start).replace(tzinfo=NY)
    right = datetime.fromisoformat(end).replace(tzinfo=NY) + timedelta(days=1)
    if right > datetime.now(timezone.utc):
        raise ValueError('End must be a completed New York date')
    for folder in ['data/processed', 'outputs', 'data/raw/text']:
        (ROOT/folder).mkdir(parents=True, exist_ok=True)
    probes = access_probes(left, right)
    polling_benchmark()
    jobs = []
    day = left
    while day < right:
        jobs.extend(('rss', (day, family)) for family in NEWS_QUERIES)
        day += timedelta(days=1)
    week = left
    while week < right:
        following = min(week+timedelta(days=7), right)
        jobs.extend(('hn', (query, int(week.timestamp()), int(following.timestamp()))) for query in HN_QUERIES)
        week = following
    # One GDELT access sample each week verifies API access and first-seen metadata,
    # without combining heterogeneous title/first-index/publication timestamps.
    if gdelt:
        jobs.extend(('gdelt', (day.astimezone(timezone.utc),)) for day in pd.date_range(left, right-timedelta(days=1), freq='7D').to_pydatetime())
    rows, audit, supplement = [], [], []
    dispatch = {'rss':rss_job, 'hn':hn_shard, 'gdelt':gdelt_job}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(dispatch[kind], *args):kind for kind, args in jobs}
        for n, future in enumerate(as_completed(futures), 1):
            found, records = future.result()
            (supplement if futures[future]=='gdelt' else rows).extend(found)
            audit.extend(records)
            if n % 20 == 0:
                print(f'Text requests completed {n}/{len(jobs)}; {len(rows)} candidate hit instances', flush=True)
    if gdelt and not supplement:
        supplement, access_audit = small_gdelt_probe()
        audit.append(access_audit)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError('No text records returned; inspect raw API responses')
    frame = frame.sort_values(['retrieved_at','id','query']).drop_duplicates('id', keep='first')
    candidates = []
    seen = set()
    for row in frame.to_dict('records'):
        row['exclusion'] = screen(row, left, right)
        norm = re.sub(r'\W+', ' ', row['text'].lower()).strip()
        row['text_sha256'] = hashlib.sha256(norm.encode()).hexdigest()
        # One headline per normalized title across outlets/queries; separate social
        # comments with identical own text also count only once.
        key = (row['source_type'], norm)
        if not row['exclusion'] and key in seen:
            row['exclusion'] = 'duplicate normalized own text'
        if not row['exclusion']:
            seen.add(key)
        try:
            row['date'] = pd.Timestamp(row['published_at']).tz_convert('America/New_York').date().isoformat()
        except (ValueError, TypeError, AttributeError):
            row['date'] = ''
        candidates.append(row)
    candidates = pd.DataFrame(candidates)
    included = candidates[candidates.exclusion.eq('')].sort_values(['published_at','source_type','id'])
    candidates.to_csv(ROOT/'data/processed/text_candidates.csv', index=False)
    included.to_csv(ROOT/'data/processed/texts.csv', index=False)
    pd.DataFrame(supplement).to_csv(ROOT/'data/processed/gdelt_supplement.csv', index=False)
    pd.DataFrame(audit).to_csv(ROOT/'outputs/text_request_audit.csv', index=False)
    pd.DataFrame(probes).to_csv(ROOT/'outputs/text_access_probes.csv', index=False)
    counts = included.groupby(['date','source_type']).size().unstack(fill_value=0)
    counts = counts.reindex(pd.date_range(start,end).strftime('%Y-%m-%d'), fill_value=0)
    counts.rename_axis('date').to_csv(ROOT/'outputs/text_daily_coverage.csv')
    summary = dict(requested_start_et=left.isoformat(), requested_end_exclusive_et=right.isoformat(),
        collected_at=utcnow(), cache_only_run=offline, news_queries=NEWS_QUERIES, hn_queries=HN_QUERIES,
        unique_candidates=len(candidates), included=len(included),
        by_source_type=included.source_type.value_counts().to_dict(), by_provider=included.provider.value_counts().to_dict(),
        exclusions=candidates.exclusion.value_counts().to_dict(),
        source_dates=included.groupby('source_type').date.agg(['min','max','nunique']).to_dict('index'),
        training_through_aug21=included[included.date.le('2026-08-21')].source_type.value_counts().to_dict(),
        requests=len(audit), capped_requests=sum(bool(r.get('capped')) for r in audit),
        gdelt_supplement_articles=len(supplement), access_probes=probes,
        historical_vintage='Retrospective provider timestamps and text as retrieved; historical original-vintage text unavailable',
        sample_limit='Ranked capped RSS headline discovery plus keyword-selected HN technology community; neither source represents US voters',
        availability='RSS next New York midnight after publication; HN own creation timestamp. Both are retrospective publication proxies, not actual historical ingestion proof.')
    (ROOT/'outputs/text_collection_summary.json').write_text(json.dumps(summary,indent=2), encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ['included','by_source_type','source_dates','training_through_aug21','capped_requests']},indent=2),flush=True)
    return included


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2026-08-01')
    parser.add_argument('--end', default='2026-10-08', help='inclusive completed New York calendar date')
    parser.add_argument('--workers', default=6, type=int)
    parser.add_argument('--no-gdelt', action='store_true')
    parser.add_argument('--offline', action='store_true', help='Rebuild only from cached payloads; missing caches raise an error')
    args = parser.parse_args()
    collect(args.start, args.end, args.workers, not args.no_gdelt, args.offline)


if __name__ == '__main__':
    main()
