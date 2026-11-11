import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# =========================
# إعداد الصفحة
# =========================

st.set_page_config(
    page_title="Crypto AI Pro",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Crypto AI Pro")
st.subheader("ماسح وتحليل العقود الآجلة للعملات الرقمية")
st.caption("تحليل فني تعليمي — ليس توصية مالية أو ضماناً للربح.")

# =========================
# منصات العقود الآجلة فقط
# =========================

EXCHANGES = {
    "Binance Futures": ccxt.binanceusdm,
    "Bybit Futures": ccxt.bybit,
    "OKX Futures": ccxt.okx,
    "Bitget Futures": ccxt.bitget,
    "KuCoin Futures": ccxt.kucoinfutures,
}

TIMEFRAMES = [
    "1m",
    "5m",
    "15m",
    "1h",
    "4h",
    "1d",
    "1w"
]


def make_exchange(name):
    exchange_class = EXCHANGES[name]

    options = {
        "enableRateLimit": True,
        "timeout": 20000,
    }

    if name == "Binance Futures":
        options["options"] = {
            "defaultType": "future"
        }
    else:
        options["options"] = {
            "defaultType": "swap"
        }

    return exchange_class(options)


# =========================
# تحديد رمز العملة
# =========================

def resolve_symbol(exchange, base):
    markets = exchange.load_markets()

    base = base.upper().strip()

    candidates = []

    for symbol, market in markets.items():

        if market.get("active") is False:
            continue

        if market.get("base") != base:
            continue

        if market.get("quote") != "USDT":
            continue

        if not (
            market.get("swap")
            or market.get("future")
        ):
            continue

        if market.get("linear") is False:
            continue

        candidates.append((symbol, market))

    for symbol, market in candidates:
        if market.get("swap") and market.get("settle") == "USDT":
            return symbol

    for symbol, market in candidates:
        if market.get("settle") == "USDT":
            return symbol

    if candidates:
        return candidates[0][0]

    raise ValueError(
        f"لم يتم العثور على عقد آجل USDT للعملة {base}."
    )


# =========================
# جلب بيانات الشموع
# =========================

def candles(exchange, symbol, timeframe, limit=250):

    rows = exchange.fetch_ohlcv(
        symbol,
        timeframe=timeframe,
        limit=limit
    )

    if not rows:
        raise ValueError("لم تصل بيانات الشموع.")

    df = pd.DataFrame(
        rows,
        columns=[
            "time",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    )

    df["time"] = pd.to_datetime(
        df["time"],
        unit="ms",
        utc=True
    )

    return df


# =========================
# مؤشر RSI
# =========================

def rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0).ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    loss = (-delta.clip(upper=0)).ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = gain / loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


# =========================
# التحليل الفني
# =========================

def analyze(df):

    if len(df) < 210:
        raise ValueError("عدد الشموع غير كافٍ للتحليل.")

    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    ema20 = c.ewm(
        span=20,
        adjust=False
    ).mean()

    ema50 = c.ewm(
        span=50,
        adjust=False
    ).mean()

    ema200 = c.ewm(
        span=200,
        adjust=False
    ).mean()

    rsi_value = rsi(c)

    macd = (
        c.ewm(span=12, adjust=False).mean()
        -
        c.ewm(span=26, adjust=False).mean()
    )

    signal = macd.ewm(
        span=9,
        adjust=False
    ).mean()

    middle = c.rolling(20).mean()

    previous_close = c.shift(1)

    true_range = pd.concat(
        [
            h - l,
            (h - previous_close).abs(),
            (l - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    atr = true_range.rolling(14).mean()

    # استخدام آخر شمعة مغلقة لتقليل تغير الإشارة أثناء تكون الشمعة
    i = len(df) - 2

    price = float(c.iloc[i])

    current_atr = float(atr.iloc[i])

    if not np.isfinite(current_atr) or current_atr <= 0:
        current_atr = price * 0.01

    current_rsi = float(rsi_value.iloc[i])

    if not np.isfinite(current_rsi):
        current_rsi = 50

    score_long = 0
    score_short = 0

    # السعر مقابل EMA20
    score_long += 20 if price > ema20.iloc[i] else 0
    score_short += 20 if price < ema20.iloc[i] else 0

    # EMA20 مقابل EMA50
    score_long += 15 if ema20.iloc[i] > ema50.iloc[i] else 0
    score_short += 15 if ema20.iloc[i] < ema50.iloc[i] else 0

    # EMA50 مقابل EMA200
    score_long += 15 if ema50.iloc[i] > ema200.iloc[i] else 0
    score_short += 15 if ema50.iloc[i] < ema200.iloc[i] else 0

    # MACD
    score_long += 15 if macd.iloc[i] > signal.iloc[i] else 0
    score_short += 15 if macd.iloc[i] < signal.iloc[i] else 0

   
