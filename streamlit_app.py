from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from datetime import time as dt_time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from model import Kronos, KronosPredictor, KronosTokenizer

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

SYSTEM_NAME = "美股监控系统 V2 云端版"
INTERVAL = "5m"
DOWNLOAD_PERIOD = "10d"
LOOKBACK = 512
PRED_LEN = 78
MARKET_TIMEZONE = "America/New_York"
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_NAME = "NeoQuasar/Kronos-small"
TOKENIZER_ASSET = "Kronos-Tokenizer-base.model.safetensors"
MODEL_ASSET = "Kronos-small.model.safetensors"

DEFAULT_TICKERS = [
    "QQQ", "NVDA", "MSFT", "UNH", "NEE", "CBRS", "INTC", "NVT", "MRVL", "VRT",
    "AVGO", "ANET", "ETN", "ABB", "NOK", "AMAT", "MU", "TSLA", "DLR", "RKLB",
]
HOLDINGS = {
    "NVDA": {"shares": 4.0021, "cost": 212.25, "rule": "不继续补仓；AI主线强则持有；跌回220下方且15-30分钟收不回或跌破218放量，卖1股机动仓。"},
    "MSFT": {"shares": 1.0, "cost": None, "rule": "长期持有，不主动加仓；AI Coding新闻是长期利好但不是单独追高信号。"},
    "UNH": {"shares": 1.0, "cost": None, "rule": "372-375观察补仓；370以下风险观察；医保/政策/司法/业绩负面新闻优先复核。"},
    "NEE": {"shares": 4.0, "cost": 85.0, "rule": "长期收息仓，不频繁做T；84以下重新评估；跌破80收不回进入风控讨论。"},
    "CBRS": {"shares": 1.0, "cost": None, "rule": "高波动主题票，不当核心仓；248-252才考虑买回T仓；270以上不追。"},
    "INTC": {"shares": 3.0, "cost": 112.0, "rule": "重点复核官方新闻、财报、代工、补贴、订单、资本开支和毛利率；不因单日新闻追高。"},
    "NVT": {"shares": 3.0, "cost": 174.0, "rule": "AI电力链重点持仓；站稳VWAP且链路强可持有；跌破买入价且链路转弱需复核。"},
}
DOWNGRADED = {"DXYZ", "RKLB", "ASTS", "LUNR", "RDW", "CBRS", "FLNC"}
THEME_CHAINS = {
    "AI核心算力链": ["NVDA", "AVGO", "AMD", "TSM", "ARM", "QCOM"],
    "AI电力链": ["VRT", "NVT", "ETN", "ABB", "GEV", "MOD", "FLNC", "GNRC", "AAON"],
    "AI网络/光互联链": ["MRVL", "AVGO", "ANET", "NOK", "COHR", "ALAB"],
    "AI软件/Agent链": ["NOW", "ADBE", "PLTR", "MSFT", "CRM", "CRWD", "PANW"],
    "AI服务器链": ["DELL", "HPE", "SMCI", "CLS", "FLEX", "JBL"],
    "半导体设备/存储": ["AMAT", "MU", "ASML", "LRCX"],
    "数据中心REIT": ["DLR", "EQIX"],
    "卖电/电力供应": ["NEE", "CEG", "VST", "NRG"],
    "太空/国防链": ["RKLB", "ASTS", "LUNR", "RDW", "DXYZ", "VSAT", "IRDM", "LMT", "RTX", "NOC"],
}
DISCIPLINE = [
    "没有三源实时确认时，结论降级为观察，不给精确挂单价。",
    "盘前暴涨票不追；高波动票最多1股试错。",
    "开盘后强势股等待15-30分钟VWAP确认；跌破VWAP且15分钟收不回，取消试错。",
    "价格远离VWAP超过5%时等待回踩，不做急拉追高。",
    "QQQ跳水或NVDA转弱时，AI链试错降级。",
]
TOKENIZER_CONFIG = {"attn_dropout_p": 0.0, "beta": 0.05, "d_in": 6, "d_model": 256, "ff_dim": 512, "ffn_dropout_p": 0.0, "gamma": 1.1, "gamma0": 1.0, "group_size": 4, "n_dec_layers": 4, "n_enc_layers": 4, "n_heads": 4, "resid_dropout_p": 0.0, "s1_bits": 10, "s2_bits": 10, "zeta": 0.05}
MODEL_CONFIG = {"attn_dropout_p": 0.1, "d_model": 512, "ff_dim": 1024, "ffn_dropout_p": 0.25, "learn_te": True, "n_heads": 8, "n_layers": 8, "resid_dropout_p": 0.25, "s1_bits": 10, "s2_bits": 10, "token_dropout_p": 0.1}

@dataclass(frozen=True)
class PredictionResult:
    ticker: str
    history: pd.DataFrame | None
    prediction: pd.DataFrame | None
    status: str
    message: str


def secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return os.environ.get(name, default)


def parse_tickers(text: str) -> list[str]:
    return [x.strip().upper() for x in text.replace("\n", ",").split(",") if x.strip()]


def chain_for(ticker: str) -> str:
    return ", ".join(name for name, tickers in THEME_CHAINS.items() if ticker in tickers)


def profile_for(ticker: str) -> dict[str, object]:
    holding = HOLDINGS.get(ticker, {})
    status = "持仓" if ticker in HOLDINGS else "观察"
    if ticker in DOWNGRADED:
        status = "降级观察"
    return {"v2_status": status, "shares": holding.get("shares"), "cost": holding.get("cost"), "chain": chain_for(ticker), "rule": holding.get("rule", "按V2纪律观察：需市场环境、新闻催化、量价状态、VWAP和QQQ相对强弱共同确认。")}


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
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep].dropna(subset=["open", "high", "low", "close"])
    df = df.reset_index().rename(columns={"Datetime": "timestamps", "Date": "timestamps"})
    if "timestamps" not in df.columns:
        df = df.rename(columns={df.columns[0]: "timestamps"})
    df["timestamps"] = pd.to_datetime(df["timestamps"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamps"]).sort_values("timestamps")
    df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0)
    avg_price = df[["open", "high", "low", "close"]].astype(float).mean(axis=1)
    df["amount"] = df["volume"].astype(float) * avg_price
    return df[["timestamps", "open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=900)
def download_market_data(tickers: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    raw = yf.download(list(tickers), period=DOWNLOAD_PERIOD, interval=INTERVAL, group_by="ticker", auto_adjust=False, progress=False, threads=True)
    for ticker in tickers:
        data[ticker] = normalize_ohlcv(raw, ticker)
    return data


def quote_from_yfinance(history: pd.DataFrame | None) -> tuple[float | None, str]:
    if history is None or history.empty:
        return None, "无 yfinance 5m 数据"
    return float(history["close"].iloc[-1]), str(history["timestamps"].iloc[-1])


def quote_from_twelvedata(ticker: str) -> tuple[float | None, str]:
    token = secret("TWELVEDATA_API_KEY")
    if not token:
        return None, "缺少 TWELVEDATA_API_KEY"
    try:
        url = "https://api.twelvedata.com/quote?" + urlencode({"symbol": ticker, "apikey": token})
        payload = requests.get(url, timeout=15).json()
        if payload.get("status") == "error":
            return None, payload.get("message", "Twelve Data error")
        return float(payload.get("close") or payload.get("price")), payload.get("datetime") or "Twelve Data"
    except Exception as exc:
        return None, str(exc)


def compare_quotes(twelve: float | None, yf_price: float | None) -> tuple[str, float | None]:
    if twelve is None or yf_price is None:
        return "不完整", None
    mid = (twelve + yf_price) / 2
    diff = abs(twelve - yf_price) / mid * 100 if mid else None
    if diff is not None and diff <= 0.3:
        return "双源一致", diff
    if diff is not None and diff <= 1.0:
        return "轻微分歧", diff
    return "明显分歧", diff


def load_state_dict(path: str) -> dict:
    state = {}
    with safe_open(path, framework="pt", device="cpu") as tensors:
        for key in tensors.keys():
            state[key] = tensors.get_tensor(key)
    return state


def model_weight(asset: str, repo_id: str) -> str:
    base_url = secret("KRONOS_MODEL_RELEASE_BASE_URL").rstrip("/")
    cache = Path(tempfile.gettempdir()) / "kronos_models"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / asset
    if target.exists() and target.stat().st_size > 0:
        return str(target)
    if base_url:
        urlretrieve(f"{base_url}/{asset}", target)
        return str(target)
    return hf_hub_download(repo_id=repo_id, filename="model.safetensors", cache_dir=cache, etag_timeout=60)


@st.cache_resource(show_spinner=False)
def load_predictor():
    tokenizer = KronosTokenizer(**TOKENIZER_CONFIG)
    tokenizer.load_state_dict(load_state_dict(model_weight(TOKENIZER_ASSET, TOKENIZER_NAME)))
    model = Kronos(**MODEL_CONFIG)
    model.load_state_dict(load_state_dict(model_weight(MODEL_ASSET, MODEL_NAME)))
    return KronosPredictor(model, tokenizer, device="cpu", max_context=LOOKBACK)


def next_regular_session_timestamps(last_timestamp: pd.Timestamp, pred_len: int = PRED_LEN) -> pd.Series:
    ts = pd.Timestamp(last_timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    local_last = ts.tz_convert(MARKET_TIMEZONE)
    next_day = (local_last + pd.Timedelta(days=1)).normalize()
    while next_day.weekday() >= 5:
        next_day += pd.Timedelta(days=1)
    start = pd.Timestamp.combine(next_day.date(), dt_time(9, 30)).tz_localize(MARKET_TIMEZONE)
    return pd.Series(pd.date_range(start=start, periods=pred_len, freq="5min").tz_convert("UTC"))


def predict_one(ticker: str, history: pd.DataFrame) -> PredictionResult:
    try:
        clean = history.dropna(subset=["timestamps", "open", "high", "low", "close"]).tail(min(LOOKBACK, len(history))).reset_index(drop=True)
        if len(clean) < 78:
            return PredictionResult(ticker, history, None, "skipped", f"有效5m K线不足：{len(clean)}")
        x_df = clean[["open", "high", "low", "close", "volume", "amount"]]
        x_ts = clean["timestamps"].dt.tz_convert("UTC").dt.tz_localize(None)
        y_ts = next_regular_session_timestamps(clean["timestamps"].iloc[-1]).dt.tz_convert("UTC").dt.tz_localize(None)
        pred = load_predictor().predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=PRED_LEN, T=1.0, top_p=0.9, sample_count=1, verbose=False)
        pred = pred.reset_index().rename(columns={"index": "timestamps"})
        pred["timestamps"] = pd.to_datetime(pred["timestamps"], utc=True, errors="coerce")
        return PredictionResult(ticker, clean, pred, "predicted", "Prediction completed")
    except Exception as exc:
        return PredictionResult(ticker, history, None, "prediction_error", str(exc))


def metrics_for(result: PredictionResult) -> dict[str, object]:
    row = {"ticker": result.ticker, "status": result.status, "message": result.message, "last_close": np.nan, "forecast_close": np.nan, "forecast_return_pct": np.nan, "forecast_max_drawdown_pct": np.nan, "forecast_volatility_pct": np.nan, "risk_label": "无信号"}
    if result.history is None or result.history.empty:
        return row
    last_close = float(result.history["close"].iloc[-1])
    row["last_close"] = last_close
    if result.prediction is None or result.prediction.empty:
        return row
    close = result.prediction["close"].astype(float)
    forecast_close = float(close.iloc[-1])
    returns = close.pct_change().dropna()
    path = pd.concat([pd.Series([last_close]), close], ignore_index=True)
    max_dd = float((path / path.cummax() - 1).min() * 100)
    vol = float(returns.std(ddof=0) * np.sqrt(len(close)) * 100) if not returns.empty else 0.0
    ret = (forecast_close / last_close - 1) * 100
    label = "高风险" if max_dd <= -4 or vol >= 5 else "偏强" if ret >= 1.5 and max_dd > -2.5 else "偏弱" if ret <= -1.5 else "中性"
    row.update({"forecast_close": forecast_close, "forecast_return_pct": ret, "forecast_max_drawdown_pct": max_dd, "forecast_volatility_pct": vol, "risk_label": label})
    return row


st.set_page_config(page_title=SYSTEM_NAME, page_icon="K", layout="wide")
st.title("美股监控系统 V2 云端版")
st.caption("Kronos 5分钟预测 + 持仓纪律雷达。没有三源实时确认时，所有动作默认降级为观察。")

with st.sidebar:
    st.subheader("股票池")
    ticker_text = st.text_area("股票代码", ", ".join(DEFAULT_TICKERS), height=130)
    prediction_limit = st.number_input("Kronos预测标的上限", min_value=0, max_value=3, value=int(secret("PREDICTION_TICKER_LIMIT", "1")), step=1)
    quote_limit = st.number_input("双源报价复核上限", min_value=1, max_value=20, value=int(secret("QUOTE_CHECK_LIMIT", "8")), step=1)
    run = st.button("开始扫描", type="primary", use_container_width=True)
    st.divider()
    st.subheader("硬规则")
    for item in DISCIPLINE:
        st.caption(f"- {item}")

if not run:
    st.info("点击“开始扫描”后下载行情并运行云端扫描。云端免费 CPU 建议每次只预测 1 个标的。")
    st.stop()

started = time.perf_counter()
tickers = parse_tickers(ticker_text)
progress = st.progress(0, text="正在下载 yfinance 5m 行情")
market_data = download_market_data(tuple(tickers))
progress.progress(35, text="正在运行 Kronos/报价复核")
results = {ticker: PredictionResult(ticker, data, None, "quote_only", "仅行情") for ticker, data in market_data.items()}
for ticker in [t for t in tickers if not market_data.get(t, pd.DataFrame()).empty][: int(prediction_limit)]:
    results[ticker] = predict_one(ticker, market_data[ticker])

rows = []
for ticker in tickers:
    result = results.get(ticker, PredictionResult(ticker, None, None, "empty", "无数据"))
    row = metrics_for(result)
    profile = profile_for(ticker)
    yf_price, yf_time = quote_from_yfinance(result.history)
    twelve_price, twelve_msg = quote_from_twelvedata(ticker) if len(rows) < int(quote_limit) else (None, "超过复核上限")
    quote_check, diff = compare_quotes(twelve_price, yf_price)
    cost = profile.get("cost")
    row.update(profile)
    row.update({"yfinance_price": yf_price, "twelvedata_price": twelve_price, "quote_check": quote_check, "quote_diff_pct": diff, "quote_note": f"yfinance={yf_time}; Twelve={twelve_msg}", "cost_distance_pct": (yf_price / cost - 1) * 100 if yf_price and cost else np.nan})
    if row["status"] != "predicted":
        row["v2_action"] = "可观察"
    elif row["v2_status"] == "降级观察":
        row["v2_action"] = "降级观察"
    elif ticker in HOLDINGS:
        row["v2_action"] = "继续持有" if row["risk_label"] != "高风险" else "止损/减仓观察"
    else:
        row["v2_action"] = "可观察"
    rows.append(row)

leaderboard = pd.DataFrame(rows)
progress.progress(100, text="完成")
time.sleep(0.2)
progress.empty()

c1, c2, c3, c4 = st.columns(4)
c1.metric("标的数量", len(tickers))
c2.metric("已预测", int((leaderboard["status"] == "predicted").sum()))
c3.metric("模式", "云端Kronos" if (leaderboard["status"] == "predicted").any() else "报价复核")
c4.metric("耗时", f"{time.perf_counter() - started:.1f} 秒")

st.subheader("V2 监控雷达")
show_cols = ["ticker", "v2_status", "v2_action", "chain", "shares", "cost", "cost_distance_pct", "twelvedata_price", "yfinance_price", "quote_check", "quote_diff_pct", "last_close", "forecast_return_pct", "forecast_max_drawdown_pct", "forecast_volatility_pct", "forecast_close", "risk_label", "status", "message", "quote_note"]
st.dataframe(leaderboard[show_cols], use_container_width=True, hide_index=True)

valid = [ticker for ticker in tickers if results.get(ticker) and results[ticker].history is not None and not results[ticker].history.empty]
selected = st.selectbox("个股详情", valid if valid else tickers)
if selected and selected in results:
    result = results[selected]
    st.write(f"**纪律规则**：{profile_for(selected)['rule']}")
    hist = result.history[["timestamps", "close"]].rename(columns={"close": "price"}).copy() if result.history is not None else pd.DataFrame()
    if not hist.empty:
        hist["series"] = "历史收盘"
        chart = hist
        if result.prediction is not None and not result.prediction.empty:
            pred = result.prediction[["timestamps", "close"]].rename(columns={"close": "price"}).copy()
            pred["series"] = "Kronos预测"
            chart = pd.concat([hist, pred], ignore_index=True)
        st.line_chart(chart, x="timestamps", y="price", color="series", height=420)
        st.bar_chart(result.history[["timestamps", "volume"]], x="timestamps", y="volume", height=180)
        if result.prediction is None:
            st.warning(result.message)
