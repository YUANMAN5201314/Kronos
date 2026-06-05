from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus, urlencode
from xml.etree import ElementTree

import pandas as pd
import requests
import yfinance as yf

TICKERS = [
    "QQQ", "NVDA", "MSFT", "UNH", "NEE", "CBRS", "INTC", "NVT", "MRVL", "VRT",
    "AVGO", "ANET", "ETN", "ABB", "NOK", "AMAT", "MU", "TSLA", "DLR", "RKLB",
]
QUOTE_LIMIT = int(os.environ.get("QUOTE_CHECK_LIMIT", "30"))
OUTPUT_PATH = Path("scan_results/latest.json")


def normalize_ohlcv(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(0):
            df = df[ticker]
        elif ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1)
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    keep = [col for col in ["open", "high", "low", "close", "volume"] if col in df.columns]
    if not keep:
        return pd.DataFrame()
    df = df[keep].dropna(subset=["open", "high", "low", "close"])
    if df.empty:
        return df
    df = df.reset_index().rename(columns={"Datetime": "timestamps", "Date": "timestamps"})
    if "timestamps" not in df.columns:
        df = df.rename(columns={df.columns[0]: "timestamps"})
    df["timestamps"] = pd.to_datetime(df["timestamps"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamps"]).sort_values("timestamps")
    df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0)
    return df[["timestamps", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def vwap_features(df: pd.DataFrame) -> dict[str, object]:
    if df.empty:
        return {"vwap": None, "price_vs_vwap_pct": None, "vwap_hold_15m": False}
    typical = (df["high"].astype(float) + df["low"].astype(float) + df["close"].astype(float)) / 3.0
    volume = df["volume"].astype(float).clip(lower=0)
    cvol = volume.cumsum()
    vwap = ((typical * volume).cumsum() / cvol.where(cvol != 0)).dropna()
    if vwap.empty:
        return {"vwap": None, "price_vs_vwap_pct": None, "vwap_hold_15m": False}
    close = df["close"].astype(float)
    latest_vwap = float(vwap.iloc[-1])
    latest_close = float(close.iloc[-1])
    last = pd.DataFrame({"close": close.tail(3), "vwap": vwap.tail(3)})
    hold = bool(len(last) >= 3 and (last["close"] >= last["vwap"]).all())
    return {
        "vwap": latest_vwap,
        "price_vs_vwap_pct": (latest_close / latest_vwap - 1.0) * 100.0 if latest_vwap else None,
        "vwap_hold_15m": hold,
    }


def quote_from_twelvedata(ticker: str) -> dict[str, object]:
    token = os.environ.get("TWELVEDATA_API_KEY")
    if not token:
        return {"price": None, "status": "missing_key", "timestamp": None}
    try:
        url = "https://api.twelvedata.com/quote?" + urlencode({"symbol": ticker, "apikey": token})
        payload = requests.get(url, timeout=15).json()
        if payload.get("status") == "error":
            return {"price": None, "status": payload.get("message", "error"), "timestamp": None}
        return {
            "price": float(payload.get("close") or payload.get("price")),
            "status": "ok",
            "timestamp": payload.get("datetime") or payload.get("timestamp"),
        }
    except Exception as exc:
        return {"price": None, "status": str(exc), "timestamp": None}


def news_titles(ticker: str) -> list[dict[str, str]]:
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quote_plus(ticker)}&region=US&lang=en-US"
    try:
        text = requests.get(url, timeout=12, headers={"User-Agent": "kronos-us-scanner/1.0"}).text
        root = ElementTree.fromstring(text)
    except Exception:
        return []
    rows = []
    for item in root.findall(".//item")[:3]:
        title = item.findtext("title") or ""
        link = item.findtext("link") or ""
        if title:
            rows.append({"source": "Yahoo Finance", "title": title, "url": link})
    return rows


def main() -> None:
    raw = yf.download(TICKERS, period="10d", interval="5m", group_by="ticker", auto_adjust=False, progress=False, threads=True)
    rows = []
    for index, ticker in enumerate(TICKERS):
        hist = normalize_ohlcv(raw, ticker)
        yf_price = None if hist.empty else float(hist["close"].iloc[-1])
        previous = None if len(hist) < 2 else float(hist["close"].iloc[-2])
        change_pct = (yf_price / previous - 1.0) * 100.0 if yf_price and previous else None
        twelve = quote_from_twelvedata(ticker) if index < QUOTE_LIMIT else {"price": None, "status": "over_limit", "timestamp": None}
        rows.append(
            {
                "ticker": ticker,
                "yfinance_price": yf_price,
                "twelvedata_price": twelve["price"],
                "twelvedata_status": twelve["status"],
                "change_pct": change_pct,
                "last_timestamp": None if hist.empty else str(hist["timestamps"].iloc[-1]),
                "news": news_titles(ticker),
                **vwap_features(hist),
            }
        )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(), "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
