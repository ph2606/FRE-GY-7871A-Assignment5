"""Public prediction-market and adjusted-equity acquisition with frozen provenance.

All probabilities are direct joint House/Senate contracts. No chamber independence
assumption, interpolation, future observation, or price forward filling is used.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
TZ = "America/New_York"
START = "2026-08-01"
END = "2026-10-08"
ESTIMATION_START = "2026-05-01"
POLY_SLUG = "balance-of-power-2026-midterms"
KALSHI_SERIES = "KXBALANCEPOWERCOMBO"
KALSHI_TICKER = "KXBALANCEPOWERCOMBO-27FEB-DD"
KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
SOLAR = ["FSLR", "ENPH", "RUN"]
DETENTION = ["GEO", "CXW"]
BASKET = SOLAR + DETENTION
DIAGNOSTIC = ["UNH"]
BENCHMARKS = ["SPY", "TAN", "IWM", "XLV"]
# Fully enumerated regular NYSE holidays in the explicitly scoped calendar year.
HOLIDAYS_2026 = ["2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03",
                 "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07",
                 "2026-11-26", "2026-12-25"]


def _json_write(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                               allow_nan=False), encoding="utf-8")


def cached_get(path: Path, url: str, params: dict | None = None,
               refresh: bool = False) -> tuple[dict | list, dict]:
    """Cache exact response bytes; hash-verify cached payloads on every read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = path.with_suffix(".metadata.json")
    if refresh or not path.exists():
        for attempt in range(4):
            try:
                response = requests.get(url, params=params,
                    headers={"User-Agent": "Mozilla/5.0 (academic NLP study)"},
                    timeout=60)
                response.raise_for_status()
                response.json()  # Validate before replacing a usable cache.
                break
            except (requests.RequestException, ValueError):
                if attempt == 3:
                    raise
                time.sleep(1 + attempt)
        if path.exists():
            versions = path.parent / "versions"
            versions.mkdir(exist_ok=True)
            old_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
            (versions / f"{path.stem}_{old_hash}.json").write_bytes(path.read_bytes())
            if metadata_path.exists():
                (versions / f"{path.stem}_{old_hash}.metadata.json").write_bytes(
                    metadata_path.read_bytes())
        path.write_bytes(response.content)
        _json_write(metadata_path, {
            "url": response.url, "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "sha256": hashlib.sha256(response.content).hexdigest(),
            "http_status": response.status_code, "bytes": len(response.content),
            "source_date_header": response.headers.get("Date"),
        })
    raw = path.read_bytes()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if hashlib.sha256(raw).hexdigest() != metadata["sha256"]:
        raise ValueError(f"Cache hash mismatch: {path}")
    metadata["cache_file"] = str(path.relative_to(ROOT)).replace("\\", "/")
    return json.loads(raw), metadata


def validate_window(start: str, end: str) -> pd.Timestamp:
    now = pd.Timestamp.now(tz=TZ)
    if pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError("Study start exceeds end.")
    if pd.Timestamp(end).date() >= now.date():
        raise ValueError("Study end must be a completed local calendar day.")
    if pd.Timestamp(start).year != 2026 or pd.Timestamp(end).year != 2026:
        raise ValueError("The explicitly audited NYSE holiday calendar covers 2026.")
    return now


def nyse_sessions(start: str, end: str) -> pd.DatetimeIndex:
    dates = pd.bdate_range(start, end)
    return dates[~dates.isin(pd.to_datetime(HOLIDAYS_2026))]


def daily_probability(history: pd.DataFrame, start: str, end: str,
                      hour: int = 16, max_age_hours: float = 24) -> pd.DataFrame:
    """Last observed value at or before cutoff, discarded once >24 hours old."""
    dates = pd.date_range(start, end)
    target = pd.DataFrame({"date": dates,
        "cutoff_at": dates.tz_localize(TZ) + pd.Timedelta(hours=hour)})
    output = target.copy()
    for platform, group in history.groupby("platform"):
        values = group.dropna(subset=["probability"]).sort_values("date_time")
        selected = pd.merge_asof(target.sort_values("cutoff_at"),
            values[["date_time", "probability"]], left_on="cutoff_at",
            right_on="date_time", direction="backward")
        age = (selected.cutoff_at - selected.date_time).dt.total_seconds() / 3600
        output[f"{platform}_probability"] = selected.probability.where(age.le(max_age_hours))
        output[f"{platform}_observation_at"] = selected.date_time
        output[f"{platform}_age_hours"] = age
    return output


def collect_prediction_markets(start: str = START, end: str = END,
                               refresh: bool = False) -> pd.DataFrame:
    validate_window(start, end)
    raw = ROOT / "data/raw/markets"
    processed = ROOT / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    provenance = []
    events, meta = cached_get(raw / "polymarket_event.json",
        "https://gamma-api.polymarket.com/events", {"slug": POLY_SLUG}, refresh)
    provenance.append(meta)
    event = events[0]
    market = next(m for m in event["markets"]
                  if m["question"] == "2026 Balance of Power: D Senate, D House")
    outcomes = json.loads(market["outcomes"])
    token = json.loads(market["clobTokenIds"])[outcomes.index("Yes")]
    # Weekly windows avoid the API's explicit-range maximum. Fetch one prior day
    # solely to make the first study cutoff's staleness auditable.
    begin = pd.Timestamp(start, tz=TZ) - pd.Timedelta(days=1)
    stop = pd.Timestamp(end, tz=TZ) + pd.Timedelta(days=1)
    chunks = []
    cursor = begin
    while cursor < stop:
        next_cursor = min(cursor + pd.Timedelta(days=7), stop)
        chunks.append((cursor, next_cursor))
        cursor = next_cursor

    def poly_chunk(bounds):
        left, right = bounds
        return cached_get(raw / f"polymarket_history_{left:%Y%m%d}_{right:%Y%m%d}.json",
            "https://clob.polymarket.com/prices-history",
            {"market": token, "startTs": int(left.timestamp()),
             "endTs": int(right.timestamp()), "fidelity": 60}, refresh)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(poly_chunk, chunks))
    poly_rows = []
    for payload, meta in results:
        provenance.append(meta)
        poly_rows.extend(payload.get("history", []))
    if not poly_rows:
        raise RuntimeError("Direct joint Polymarket price history was empty.")
    poly = pd.DataFrame(poly_rows).rename(columns={"t": "timestamp", "p": "probability"})
    poly["date_time"] = pd.to_datetime(poly.timestamp, unit="s", utc=True).dt.tz_convert(TZ)
    poly["platform"] = "polymarket"
    poly["measure"] = "CLOB historical YES sampled price"
    histories = [poly[["date_time", "probability", "platform", "measure"]]]
    kalshi_status = {"available": False}
    try:
        kalshi_event, meta = cached_get(raw / "kalshi_event.json", KALSHI_API + "/events",
            {"series_ticker": KALSHI_SERIES, "with_nested_markets": "true"}, refresh)
        provenance.append(meta)
        kalshi_market = next(m for e in kalshi_event["events"] for m in e["markets"]
                             if m["ticker"] == KALSHI_TICKER)
        if "House Control: Democratic, Senate Control: Democratic" not in kalshi_market["rules_primary"]:
            raise ValueError("Joint-control definition no longer matches audited rules.")
        payload, meta = cached_get(raw / f"kalshi_hourly_{start}_{end}.json",
            f"{KALSHI_API}/series/{KALSHI_SERIES}/markets/{KALSHI_TICKER}/candlesticks",
            {"start_ts": int(begin.timestamp()), "end_ts": int(stop.timestamp()),
             "period_interval": 60}, refresh)
        provenance.append(meta)
        rows = []
        for candle in payload["candlesticks"]:
            bid = candle.get("yes_bid", {}).get("close_dollars")
            ask = candle.get("yes_ask", {}).get("close_dollars")
            if bid is None or ask is None:
                continue
            bid, ask = float(bid), float(ask)
            if not 0 <= bid <= ask <= 1:
                continue
            rows.append({"date_time": pd.Timestamp(candle["end_period_ts"], unit="s", tz="UTC").tz_convert(TZ),
                "probability": (bid + ask) / 2, "platform": "kalshi",
                "measure": "hourly closing YES bid-ask midpoint", "bid": bid, "ask": ask,
                "trade_close": candle.get("price", {}).get("close_dollars"),
                "volume": float(candle.get("volume_fp", "0"))})
        kalshi = pd.DataFrame(rows)
        histories.append(kalshi)
        kalshi_status = {"available": len(kalshi) > 0, "ticker": KALSHI_TICKER,
            "rules_primary": kalshi_market["rules_primary"],
            "resolution_difference": "Kalshi control for February 2027; Polymarket November election seat attribution with specified independent-caucus and organization rules."}
    except (requests.RequestException, KeyError, StopIteration, ValueError) as exc:
        kalshi_status["error"] = f"{type(exc).__name__}: {exc}"
    history = pd.concat(histories, ignore_index=True)
    history = history.drop_duplicates(["platform", "date_time"]).sort_values(["platform", "date_time"])
    history = history.loc[(history.date_time >= begin) & (history.date_time < stop)]
    if not history.probability.between(0, 1).all():
        raise ValueError("Prediction prices outside [0,1].")
    history.to_csv(processed / "prediction_history.csv", index=False)
    daily = daily_probability(history, start, end)
    # Calendar midnight is supplied for optional EOD sensitivity only. Canonical
    # correlations use 16:00 ET to share the stock-market close information set.
    eod = daily_probability(history, start, end, hour=24)
    for name in ["polymarket", "kalshi"]:
        if f"{name}_probability" in eod:
            daily[f"{name}_probability_eod"] = eod[f"{name}_probability"]
    daily.to_csv(processed / "prediction_market_daily.csv", index=False)
    _json_write(processed / "prediction_market_manifest.json", {
        "start": start, "end": end, "as_of_local": str(pd.Timestamp.now(tz=TZ)),
        "joint_contract": {"event_slug": POLY_SLUG, "market_id": market["id"],
            "condition_id": market["conditionId"], "yes_token_id": token,
            "question": market["question"], "resolution_rules": market["description"]},
        "kalshi": kalshi_status,
        "units": "probability-equivalent price, 0 to 1; not a calibrated population probability",
        "alignment": "latest observation <=16:00 America/New_York each calendar day; maximum age 24 hours; no future interpolation",
        "polymarket_measure": "historical YES price sampled at requested hourly fidelity; API does not guarantee it is a trade or executable quote",
        "kalshi_measure": "closing YES bid-ask midpoint, excluding crossed/missing quotes; quote age within each hour not independently observable",
        "metadata_vintage": "retrieval-day metadata; historical prices individually time-filtered",
        "sources": provenance,
        "observations": {str(k): int(v) for k, v in history.groupby("platform").size().items()},
        "daily_complete": {c: int(daily[c].notna().sum()) for c in daily if c.endswith("_probability")},
    })
    return daily


def collect_equities(start: str = START, end: str = END,
                     refresh: bool = False) -> pd.DataFrame:
    now = validate_window(start, end)
    raw = ROOT / "data/raw/equities"
    processed = ROOT / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    # Record design before examining returns; this is a retrospective researcher
    # design lock, not a claim of an August 1 preregistration.
    design = {"created_at": datetime.now(timezone.utc).isoformat(),
        "study_start": start, "study_end": end, "solar": SOLAR, "detention": DETENTION,
        "long_only_weights": {t: 1 / len(BASKET) for t in BASKET},
        "signed_weights": {**{t: 0.5 / len(SOLAR) for t in SOLAR},
                           **{t: -0.5 / len(DETENTION) for t in DETENTION}},
        "diagnostic_only": DIAGNOSTIC, "benchmarks": BENCHMARKS,
        "design_status": "selected for policy mechanisms before this acquisition's return inspection; retrospective, not prospectively preregistered"}
    design_path = raw / "basket_design.json"
    if not design_path.exists():
        _json_write(design_path, design)
    begin = int(pd.Timestamp(ESTIMATION_START, tz=TZ).timestamp())
    stop = int((pd.Timestamp(end, tz=TZ) + pd.Timedelta(days=1)).timestamp())

    def one(symbol):
        payload, meta = cached_get(raw / f"{symbol}_{ESTIMATION_START}_{end}.json",
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
            {"period1": begin, "period2": stop, "interval": "1d", "events": "div,splits"}, refresh)
        result = payload["chart"]["result"][0]
        frame = pd.DataFrame({
            "date": pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert(TZ).normalize().tz_localize(None),
            "ticker": symbol,
            "adjusted_close": result["indicators"]["adjclose"][0]["adjclose"],
            "close": result["indicators"]["quote"][0]["close"],
            "volume": result["indicators"]["quote"][0]["volume"],
        })
        close_times = frame.date.dt.tz_localize(TZ) + pd.Timedelta(hours=16)
        frame = frame.loc[frame.date.le(end) & close_times.le(now)].copy()
        if frame.date.duplicated().any() or not frame.adjusted_close.dropna().gt(0).all():
            raise ValueError(f"Invalid adjusted prices: {symbol}")
        meta.update(ticker=symbol, rows=len(frame), first=str(frame.date.min().date()),
                    last=str(frame.date.max().date()))
        return frame, meta

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one, BASKET + DIAGNOSTIC + BENCHMARKS))
    levels = pd.concat([r[0] for r in results], ignore_index=True).sort_values(["date", "ticker"])
    levels.to_csv(processed / "market_levels.csv", index=False)
    sessions = nyse_sessions(ESTIMATION_START, end)
    wide = levels.pivot(index="date", columns="ticker", values="adjusted_close")
    if len(wide.index.difference(sessions)):
        raise ValueError("Feed contains dates outside the audited cash calendar.")
    # Restore missing sessions before pct_change; never bridge a missing close.
    wide = wide.reindex(sessions)
    wide.index.name = "date"
    returns = 100 * wide.pct_change(fill_method=None)
    estimation = returns.loc[returns.index < pd.Timestamp(start)]
    betas = []
    output = []
    for symbol in BASKET + DIAGNOSTIC:
        fit = estimation[[symbol, "SPY"]].dropna()
        if len(fit) < 40:
            raise ValueError(f"Insufficient strictly prewindow beta-estimation sessions: {symbol}")
        alpha, beta = np.linalg.lstsq(np.column_stack([np.ones(len(fit)), fit.SPY]), fit[symbol], rcond=None)[0]
        sector = "TAN" if symbol in SOLAR else "XLV" if symbol == "UNH" else "IWM"
        betas.append({"ticker": symbol, "alpha_pct": float(alpha), "beta_spy": float(beta),
                      "n_estimation": len(fit), "first": str(fit.index.min().date()),
                      "last": str(fit.index.max().date()), "sector_proxy": sector})
        output.append(pd.DataFrame({"date": returns.index, "ticker": symbol,
            "return_pct": returns[symbol].to_numpy(), "spy_return_pct": returns.SPY.to_numpy(),
            "market_adjusted_return_pct": (returns[symbol] - returns.SPY).to_numpy(),
            "beta_adjusted_return_pct": (returns[symbol] - beta * returns.SPY).to_numpy(),
            "market_model_residual_pct": (returns[symbol] - alpha - beta * returns.SPY).to_numpy(),
            "sector_proxy": sector, "sector_adjusted_return_pct": (returns[symbol] - returns[sector]).to_numpy(),
            "beta_spy": beta}))
    equity = pd.concat(output, ignore_index=True)
    equity = equity.loc[equity.date.between(start, end)]
    equity.to_csv(processed / "equity_returns.csv", index=False)
    beta_frame = pd.DataFrame(betas)
    beta_frame.to_csv(processed / "prewindow_betas.csv", index=False)
    basket = pd.DataFrame(index=returns.loc[start:end].index)
    for field, label in [("return_pct", "return"), ("market_adjusted_return_pct", "market_adjusted"),
                         ("beta_adjusted_return_pct", "beta_adjusted"),
                         ("market_model_residual_pct", "market_model_residual")]:
        values = equity.pivot(index="date", columns="ticker", values=field)
        basket[f"long_only_{label}_pct"] = values[BASKET].mean(axis=1, skipna=False)
        basket[f"signed_{label}_pct"] = (0.5 * values[SOLAR].mean(axis=1, skipna=False)
                                          - 0.5 * values[DETENTION].mean(axis=1, skipna=False))
    basket["spy_return_pct"] = returns.SPY
    basket["tan_return_pct"] = returns.TAN
    basket["iwm_return_pct"] = returns.IWM
    basket["xlv_return_pct"] = returns.XLV
    # Analysis-facing aliases deliberately use decimal returns. Explicit *_pct
    # columns above remain available for readable tables and unit checking.
    basket["solar_raw"] = returns[SOLAR].mean(axis=1, skipna=False) / 100
    basket["detention_raw"] = returns[DETENTION].mean(axis=1, skipna=False) / 100
    basket["spy"] = returns.SPY / 100
    basket["solar_minus_spy"] = basket.solar_raw - basket.spy
    basket["detention_minus_spy"] = basket.detention_raw - basket.spy
    basket["signed_raw"] = basket.signed_return_pct / 100
    basket["signed_beta_adjusted"] = basket.signed_beta_adjusted_pct / 100
    basket["long_only_minus_spy"] = basket.long_only_market_adjusted_pct / 100
    for symbol in BASKET + DIAGNOSTIC:
        basket[f"{symbol.lower()}_raw"] = returns[symbol] / 100
        basket[f"{symbol.lower()}_minus_spy"] = (returns[symbol] - returns.SPY) / 100
    basket["long_only_wealth"] = (1 + basket.long_only_return_pct / 100).cumprod(skipna=False)
    basket["signed_wealth"] = (1 + basket.signed_return_pct / 100).cumprod(skipna=False)
    basket.to_csv(processed / "basket_daily.csv", index_label="date")
    _json_write(processed / "market_manifest.json", {
        **design, "sources": [r[1] for r in results],
        "return_units": "simple daily percent adjusted-close returns; beta model in identical units",
        "analysis_alias_units": "solar_raw, detention_raw, spy, *_minus_spy, signed_raw, signed_beta_adjusted, and individual *_raw are DECIMAL daily returns; all *_pct fields are percent",
        "adjusted_price_vintage": "latest retrospective adjusted close at retrieval, not point-in-time corporate-action vintage",
        "estimation": "OLS individual equity returns on intercept + SPY using May 1 through July 31, strictly before study window; coefficients frozen",
        "market_adjustment": "equity minus SPY; long-only basket minus SPY; additionally equity minus frozen beta*SPY and market-model residual with frozen intercept",
        "rebalancing": "equal-dollar weights restored daily at previous close; frictionless descriptive basket",
        "signed_net_exposure": 0, "signed_gross_exposure": 1,
        "signed_spy_cancellation": "0.5*mean(solar-SPY)-0.5*mean(detention-SPY) equals raw signed return; beta-adjusted returns address unequal market betas",
        "diagnostic_exclusion": "UNH has ambiguous net election direction (public-program funding versus insurer regulation); excluded from directional basket",
        "calendar": "2026 NYSE weekdays excluding explicitly enumerated holidays; missing close stays missing before pct_change",
        "missing_adjusted_prices": {str(t): int(v) for t, v in wide.isna().sum().items()},
        "sessions": len(basket), "start": start, "end": end,
    })
    return basket


def collect(start: str = START, end: str = END, refresh: bool = False) -> dict:
    prediction = collect_prediction_markets(start, end, refresh)
    basket = collect_equities(start, end, refresh)
    summary = {"calendar_days": len(prediction), "stock_sessions": len(basket),
               "prediction_columns": prediction.columns.tolist(), "basket_columns": basket.columns.tolist()}
    print(json.dumps(summary, indent=2))
    return summary
