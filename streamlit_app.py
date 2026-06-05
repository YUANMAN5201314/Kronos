from __future__ import annotations

import os
import time
from dataclasses import dataclass
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

SYSTEM_NAME = "美股监控系统 V2 云端版"
INTERVAL = "5m"
DOWNLOAD_PERIOD = "10d"
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

@dataclass(frozen=True)
class MarketRow:
    ticker: str
    history: pd.DataFrame | None
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
    return {
        "v2_status": status,
        "shares": holding.get("shares"),
        "cost": holding.get("cost"),
        "chain": chain_for(ticker),
        "rule": holding.get("rule", "按V2纪律观察：需市场环境、新闻催化、量价状态、VWAP和QQQ相对强弱共同确认。"),
    }


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


@st.cache_data(show_spinner=False, ttl=600)
def download_market_data(tickers: tuple[str, ...]) -> dict[str, MarketRow]:
    rows: dict[str, MarketRow] = {}
    try:
        raw = yf.download(list(tickers), period=DOWNLOAD_PERIOD, interval=INTERVAL, group_by="ticker", auto_adjust=False, progress=False, threads=True)
    except Exception as exc:
        return {ticker: MarketRow(ticker, None, f"yfinance下载失败：{exc}") for ticker in tickers}
    for ticker in tickers:
        hist = normalize_ohlcv(raw, ticker)
        msg = "行情可用" if hist is not None and not hist.empty else "无5m行情"
        rows[ticker] = MarketRow(ticker, hist, msg)
    return rows


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


def add_vwap(data: pd.DataFrame | None) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()
    df = data.copy()
    typical = (df["high"].astype(float) + df["low"].astype(float) + df["close"].astype(float)) / 3.0
    volume = df["volume"].astype(float).clip(lower=0)
    cvol = volume.cumsum()
    df["vwap"] = (typical * volume).cumsum() / cvol.where(cvol != 0)
    return df


def summarize_market(ticker: str, row: MarketRow, quote_index: int, quote_limit: int) -> dict[str, object]:
    profile = profile_for(ticker)
    yf_price, yf_note = quote_from_yfinance(row.history)
    twelve_price, twelve_note = quote_from_twelvedata(ticker) if quote_index < quote_limit else (None, "超过复核上限")
    quote_check, quote_diff = compare_quotes(twelve_price, yf_price)
    hist = add_vwap(row.history)
    vwap = float(hist["vwap"].iloc[-1]) if not hist.empty and "vwap" in hist else np.nan
    price_vs_vwap = (yf_price / vwap - 1) * 100 if yf_price and vwap == vwap and vwap else np.nan
    last3 = hist.tail(3) if not hist.empty else pd.DataFrame()
    vwap_hold_15m = bool(len(last3) >= 3 and (last3["close"].astype(float) >= last3["vwap"].astype(float)).all()) if not last3.empty else False
    prev_close = float(hist["close"].iloc[-2]) if len(hist) >= 2 else np.nan
    change_pct = (yf_price / prev_close - 1) * 100 if yf_price and prev_close == prev_close and prev_close else np.nan
    cost = profile.get("cost")
    action = "可观察"
    if profile["v2_status"] == "降级观察":
        action = "降级观察"
    elif ticker in HOLDINGS:
        action = "继续持有"
    if quote_check != "双源一致":
        action = "可观察" if action != "降级观察" else action
    return {
        "ticker": ticker,
        "v2_status": profile["v2_status"],
        "v2_action": action,
        "chain": profile["chain"],
        "shares": profile["shares"],
        "cost": cost,
        "cost_distance_pct": (yf_price / cost - 1) * 100 if yf_price and cost else np.nan,
        "yfinance_price": yf_price,
        "twelvedata_price": twelve_price,
        "quote_check": quote_check,
        "quote_diff_pct": quote_diff,
        "change_pct": change_pct,
        "vwap": vwap,
        "price_vs_vwap_pct": price_vs_vwap,
        "vwap_hold_15m": vwap_hold_15m,
        "status": row.message,
        "quote_note": f"yfinance={yf_note}; Twelve={twelve_note}",
        "discipline_rule": profile["rule"],
    }


st.set_page_config(page_title=SYSTEM_NAME, page_icon="K", layout="wide")
st.title("美股监控系统 V2 云端版")
st.caption("云端优先稳定展示：行情 + 双源报价复核 + VWAP/持仓纪律雷达。完整 Kronos 5分钟预测请使用本地版。")
st.info("云端免费环境不强制加载 Kronos/torch，避免手机网页长时间卡在依赖安装。")

with st.sidebar:
    st.subheader("股票池")
    ticker_text = st.text_area("股票代码", ", ".join(DEFAULT_TICKERS), height=130)
    quote_limit = st.number_input("双源报价复核上限", min_value=1, max_value=20, value=int(secret("QUOTE_CHECK_LIMIT", "8")), step=1)
    run = st.button("开始扫描", type="primary", use_container_width=True)
    st.divider()
    st.subheader("硬规则")
    for item in DISCIPLINE:
        st.caption(f"- {item}")

if not run:
    st.info("点击“开始扫描”后下载行情并运行云端雷达。")
    st.stop()

started = time.perf_counter()
tickers = parse_tickers(ticker_text)
progress = st.progress(0, text="正在下载 yfinance 5m 行情")
market_data = download_market_data(tuple(tickers))
progress.progress(75, text="正在做 Twelve Data / yfinance 双源报价复核")
rows = [summarize_market(ticker, market_data.get(ticker, MarketRow(ticker, None, "无数据")), idx, int(quote_limit)) for idx, ticker in enumerate(tickers)]
leaderboard = pd.DataFrame(rows)
progress.progress(100, text="完成")
time.sleep(0.2)
progress.empty()

c1, c2, c3, c4 = st.columns(4)
c1.metric("标的数量", len(tickers))
c2.metric("双源一致", int((leaderboard["quote_check"] == "双源一致").sum()))
c3.metric("模式", "云端报价雷达")
c4.metric("耗时", f"{time.perf_counter() - started:.1f} 秒")

st.subheader("V2 监控雷达")
show_cols = ["ticker", "v2_status", "v2_action", "chain", "shares", "cost", "cost_distance_pct", "twelvedata_price", "yfinance_price", "quote_check", "quote_diff_pct", "change_pct", "price_vs_vwap_pct", "vwap_hold_15m", "status", "quote_note"]
st.dataframe(leaderboard[show_cols], use_container_width=True, hide_index=True)

valid = [ticker for ticker in tickers if market_data.get(ticker) and market_data[ticker].history is not None and not market_data[ticker].history.empty]
selected = st.selectbox("个股详情", valid if valid else tickers)
if selected and selected in market_data:
    profile = profile_for(selected)
    st.write(f"**纪律规则**：{profile['rule']}")
    hist = add_vwap(market_data[selected].history)
    if not hist.empty:
        chart = hist[["timestamps", "close", "vwap"]].copy()
        chart = chart.melt(id_vars="timestamps", value_vars=["close", "vwap"], var_name="series", value_name="price")
        chart["series"] = chart["series"].map({"close": "历史收盘", "vwap": "VWAP"})
        st.line_chart(chart, x="timestamps", y="price", color="series", height=420)
        st.bar_chart(hist[["timestamps", "volume"]], x="timestamps", y="volume", height=180)
