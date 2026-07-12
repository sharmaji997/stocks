import io
import os
import pickle
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

from config import WATCHLIST
from features import FEATURE_COLUMNS, LABEL_NAMES, add_technical_indicators


_YF_SESSION = requests.Session()
_YF_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
})

MODEL_DIR = Path(__file__).parent / "model"

st.set_page_config(page_title="SignalWise", layout="wide", page_icon="📈",
                    initial_sidebar_state="expanded")

# ---------------------------------------------------------------- theme / css
# Groww-inspired light fintech palette: white surfaces, signature mint-green
# accent, warm coral for "Sell", soft amber for "Hold".
PLOTLY_TEMPLATE = "plotly_white"
ACCENT = "#00D09C"        # primary brand green
ACCENT_DEEP = "#00B386"   # darker green for text-on-white (contrast-safe)
BUY_COLOR = "#00D09C"
HOLD_COLOR = "#FFA800"
SELL_COLOR = "#EB5B3C"
BG = "#F6F8FA"
PANEL = "#FFFFFF"
BORDER = "#E7ECF1"
TEXT = "#1F2933"
SUBTEXT = "#75808C"

st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: {TEXT};
}}
.stApp {{
    background: {BG};
}}
section[data-testid="stSidebar"] {{
    background: {PANEL};
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] * {{ color: {TEXT}; }}
.desk-brand {{
    font-family: 'DM Sans', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    color: {TEXT};
    letter-spacing: -0.01em;
    padding: 0.5rem 0 0.2rem 0.6rem;
    border-left: 4px solid {ACCENT};
    margin-bottom: 0.2rem;
}}
.desk-brand span {{ color: {ACCENT_DEEP}; }}
.desk-sub {{
    color: {SUBTEXT};
    font-size: 0.82rem;
    padding-left: 0.7rem;
    margin-bottom: 1.3rem;
}}
.card {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 1px 2px rgba(16,24,40,0.04);
}}
.stat-card {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 0.95rem 1.1rem;
    text-align: left;
    box-shadow: 0 1px 2px rgba(16,24,40,0.04);
}}
.stat-label {{
    color: {SUBTEXT};
    font-size: 0.72rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}}
.stat-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: {TEXT};
    font-variant-numeric: tabular-nums;
}}
.stat-delta-up {{ color: {ACCENT_DEEP}; font-size: 0.85rem; font-family: 'JetBrains Mono', monospace; font-weight: 600; }}
.stat-delta-down {{ color: {SELL_COLOR}; font-size: 0.85rem; font-family: 'JetBrains Mono', monospace; font-weight: 600; }}
.signal-pill {{
    display: inline-block;
    font-family: 'DM Sans', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    padding: 0.35rem 1.15rem;
    border-radius: 999px;
    letter-spacing: 0.02em;
}}
.pill-buy {{ background: rgba(0,208,156,0.12); color: {ACCENT_DEEP}; border: 1px solid {ACCENT}; }}
.pill-hold {{ background: rgba(255,168,0,0.12); color: #B36B00; border: 1px solid {HOLD_COLOR}; }}
.pill-sell {{ background: rgba(235,91,60,0.10); color: {SELL_COLOR}; border: 1px solid {SELL_COLOR}; }}
.section-heading {{
    font-family: 'DM Sans', sans-serif;
    color: {TEXT};
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 1.5rem 0 0.6rem 0;
    border-left: 3px solid {ACCENT};
    padding-left: 0.6rem;
}}
.watch-row-buy {{ border-left: 3px solid {ACCENT}; }}
.watch-row-sell {{ border-left: 3px solid {SELL_COLOR}; }}
.disclaimer {{
    font-size: 0.78rem;
    color: {SUBTEXT};
    border-top: 1px solid {BORDER};
    padding-top: 0.8rem;
    margin-top: 1.5rem;
}}
div[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 10px; overflow: hidden; }}
div[data-testid="stTextInput"] input, div[data-testid="stSelectbox"] div[data-baseweb="select"] {{
    border-radius: 10px !important;
}}
.stButton > button {{
    background: {ACCENT};
    color: #FFFFFF;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.5rem 1.4rem;
}}
.stButton > button:hover {{
    background: {ACCENT_DEEP};
    color: #FFFFFF;
}}
div[role="radiogroup"] label {{
    background: {BG};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 0.45rem 0.7rem;
    margin-bottom: 0.35rem;
}}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- artifacts
@st.cache_resource
def load_artifacts():
    with open(MODEL_DIR / "signal_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODEL_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(MODEL_DIR / "feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    with open(MODEL_DIR / "metrics.pkl", "rb") as f:
        metrics = pickle.load(f)
    return model, scaler, feature_columns, metrics


_TWELVEDATA_OUTPUTSIZE = {"6mo": 130, "1y": 260, "2y": 520, "5y": 1300}


def _twelvedata_api_key():
    
    try:
        return st.secrets["TWELVEDATA_API_KEY"]
    except Exception:
        return os.environ.get("TWELVEDATA_API_KEY")


def _fetch_from_twelvedata(ticker: str, period: str) -> pd.DataFrame:
    """
    Primary data source. Twelve Data is a documented REST API with a real
    free tier (800 requests/day, 8/minute) and clear error responses --
    unlike yfinance, which scrapes Yahoo's internal endpoints and gets
    silently rate-limited/blocked with no warning.

    Coverage note: the free "Basic" plan does not include full India/NSE
    data (that's gated behind a higher paid tier), so ".NS" tickers will
    typically come back empty here and fall through to yfinance/Stooq.
    Returns an empty DataFrame (not an error) if no key is configured.
    """
    api_key = _twelvedata_api_key()
    if not api_key:
        return pd.DataFrame()

    try:
        resp = requests.get(
            "https://api.twelvedata.com/time_series",
            params={
                "symbol": ticker,
                "interval": "1day",
                "outputsize": _TWELVEDATA_OUTPUTSIZE.get(period, 260),
                "apikey": api_key,
            },
            timeout=15,
        )
        data = resp.json()
    except Exception:
        return pd.DataFrame()

    if not isinstance(data, dict) or data.get("status") == "error" or "values" not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data["values"])
    if df.empty or "datetime" not in df.columns:
        return pd.DataFrame()

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[list(required)].dropna()


_STOOQ_PERIOD_DAYS = {"6mo": 182, "1y": 365, "2y": 730, "5y": 1825}


def _fetch_from_stooq(ticker: str, period: str) -> pd.DataFrame:
    """
    Fallback data source used only when Yahoo Finance is unreachable or
    rate-limiting us. Stooq needs no API key and serves a plain CSV, so
    there's no cookie/crumb handshake for Yahoo-style blocking to break.

    Coverage note: this reliably covers US-listed tickers (Stooq wants a
    ".us" suffix, e.g. "aapl.us") but does NOT reliably cover NSE ".NS"
    tickers -- for those this will just return empty and the caller falls
    back to its normal "no data found" message.
    """
    symbol = ticker.lower()
    if not symbol.endswith(".us") and "." not in symbol:
        symbol = f"{symbol}.us"

    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    try:
        resp = _YF_SESSION.get(url, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
    except Exception:
        return pd.DataFrame()

    if df.empty or "Date" not in df.columns:
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    days = _STOOQ_PERIOD_DAYS.get(period, 365)
    cutoff = df.index.max() - pd.Timedelta(days=days)
    df = df[df.index >= cutoff]

    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    return df[list(required)]


@st.cache_data(ttl=3600)
def fetch_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    # yfinance treats whitespace inside the string as a ticker separator and
    # will silently try to download multiple symbols (e.g. "ADANI POWER.NS"
    # becomes two tickers: "ADANI" and "POWER.NS"). Strip spaces so a typo
    # like that just becomes one (possibly wrong) symbol instead of a
    # malformed multi-ticker request.
    ticker = ticker.replace(" ", "")
    if not ticker:
        return pd.DataFrame()

    # 1) Twelve Data -- documented API + published rate limits, tried first.
    df = _fetch_from_twelvedata(ticker, period)

    # 2) yfinance -- no key needed, but can be blocked/rate-limited by Yahoo.
    if df is None or df.empty:
        yf_df = None
        for attempt in range(2):
            try:
                yf_df = yf.download(ticker, period=period, progress=False, auto_adjust=True,
                                     timeout=15, session=_YF_SESSION)
            except Exception:
                yf_df = None
            if yf_df is not None and not yf_df.empty:
                break
        if yf_df is not None and not yf_df.empty:
            if isinstance(yf_df.columns, pd.MultiIndex):
                yf_df.columns = yf_df.columns.get_level_values(0)
            # Guard against duplicate column names (can still happen if
            # yfinance ever returns more than one ticker's data) -- with
            # dupes, df["Close"] returns a DataFrame instead of a Series
            # and breaks everything downstream.
            required = {"Open", "High", "Low", "Close", "Volume"}
            if not yf_df.columns.duplicated().any() and required.issubset(yf_df.columns):
                df = yf_df

    # 3) Stooq -- last resort, no key needed, mainly covers US tickers.
    if df is None or df.empty:
        df = _fetch_from_stooq(ticker, period)

    if df is None or df.empty:
        return pd.DataFrame()
    return df


def predict_signal(df: pd.DataFrame, model, scaler, feature_columns):
    """Return (label, probabilities dict) for the most recent row."""
    featurized = add_technical_indicators(df).dropna(subset=feature_columns)
    if featurized.empty:
        return None, None
    latest = featurized.iloc[[-1]][feature_columns].values
    latest_scaled = scaler.transform(latest)
    pred = model.predict(latest_scaled)[0]
    proba = model.predict_proba(latest_scaled)[0]
    proba_dict = {LABEL_NAMES[i]: float(p) for i, p in enumerate(proba)}
    return LABEL_NAMES[pred], proba_dict


def pill_class(label: str) -> str:
    return {"Buy": "pill-buy", "Hold": "pill-hold", "Sell": "pill-sell"}.get(label, "pill-hold")


def stat_card(label: str, value: str, delta: str = None, delta_up: bool = True):
    delta_html = ""
    if delta is not None:
        cls = "stat-delta-up" if delta_up else "stat-delta-down"
        arrow = "▲" if delta_up else "▼"
        delta_html = f'<div class="{cls}">{arrow} {delta}</div>'
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def themed_fig_layout(fig, height):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(family="Inter, sans-serif", color=TEXT, size=11),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.03, bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ---------------------------------------------------------------- load model
model_ready = False
model_load_error = None
try:
    model, scaler, feature_columns, metrics = load_artifacts()
    model_ready = True
except FileNotFoundError:
    model_load_error = "missing"
except Exception as exc:
    # Covers pickle/unpickling failures too -- e.g. the model artifacts were
    # saved with a different scikit-learn/Python version than what's
    # installed here. Show a clear message instead of letting the app crash.
    model_load_error = str(exc)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<div class="desk-brand">signal<span>wise</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="desk-sub">Simple stock signals, backed by data</div>',
                unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["Stock Analysis", "Watchlist Suggestions", "Model Insights"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="section-heading">Legend</div>', unsafe_allow_html=True)
    st.markdown(
        f'<span class="signal-pill pill-buy">BUY</span> &nbsp;'
        f'<span class="signal-pill pill-hold">HOLD</span> &nbsp;'
        f'<span class="signal-pill pill-sell">SELL</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="disclaimer">Educational ML project — Random Forest on '
        'technical indicators. Not financial advice.</div>',
        unsafe_allow_html=True,
    )

if not model_ready:
    if model_load_error == "missing":
        st.error(
            "No trained model found in `model/`. Run `python train_model.py` first "
            "to download data and train the model, then relaunch the app."
        )
    else:
        st.error(
            "The saved model files couldn't be loaded — this usually means they "
            "were saved with a different scikit-learn/Python version than what's "
            "installed here.\n\n"
            f"Details: {model_load_error}\n\n"
            "Fix: pin `scikit-learn` in `requirements.txt` to match the version "
            "used for training, and/or set the app's Python version (Advanced "
            "settings on Streamlit Cloud) to match, then re-run `train_model.py` "
            "and redeploy."
        )
    st.stop()

# ============================================================== PAGE: ANALYSIS
if page == "Stock Analysis":
    if "ticker_input" not in st.session_state:
        st.session_state.ticker_input = "AAPL"

    def _uppercase_ticker():
        st.session_state.ticker_input = st.session_state.ticker_input.strip().upper()

    top_l, top_r = st.columns([3, 1])
    with top_l:
        st.text_input("Ticker", key="ticker_input", on_change=_uppercase_ticker,
                       placeholder="e.g. AAPL, TSLA, RELIANCE.NS",
                       label_visibility="collapsed")
    with top_r:
        period = st.selectbox("History", ["6mo", "1y", "2y", "5y"], index=1,
                               label_visibility="collapsed")
    st.caption("NSE-listed stocks need a **.NS** suffix — e.g. `RELIANCE.NS`, `TCS.NS`, `INFY.NS`.")
    ticker = st.session_state.ticker_input.strip().upper()

    if ticker:
        with st.spinner(f"Pulling {ticker} ..."):
            try:
                hist = fetch_history(ticker, period)
            except Exception as exc:
                hist = pd.DataFrame()
                st.warning(f"Couldn't fetch data for **{ticker}**: {exc}")

        if hist.empty:
            st.warning(
                "No data found for that ticker. Check the symbol — NSE stocks "
                "need a **.NS** suffix, e.g. `RELIANCE.NS`, `TCS.NS`."
            )
        elif len(hist) < 2:
            st.warning(
                f"Only found {len(hist)} day(s) of history for **{ticker}** — "
                "try a longer period or double-check the symbol."
            )
        else:
            try:
                featurized = add_technical_indicators(hist)
                label, proba = predict_signal(hist, model, scaler, feature_columns)

                last_close = float(hist["Close"].iloc[-1])
                prev_close = float(hist["Close"].iloc[-2])
                change_pct = (last_close / prev_close - 1) * 100

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    stat_card("Last Close", f"{last_close:,.2f}", f"{change_pct:+.2f}%", change_pct >= 0)
                with c2:
                    conf = f"{max(proba.values()) * 100:.1f}%" if proba else "—"
                    stat_card("Confidence", conf)
                with c3:
                    stat_card("RSI (14)", f"{featurized['rsi_14'].iloc[-1]:.1f}")
                with c4:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-label">Model Signal</div>
                        <div style="margin-top:0.3rem;">
                            <span class="signal-pill {pill_class(label) if label else 'pill-hold'}">
                                {label.upper() if label else 'N/A'}
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                if proba:
                    st.markdown('<div class="section-heading">Signal Probabilities</div>',
                                 unsafe_allow_html=True)
                    proba_df = pd.DataFrame(
                        {"Signal": list(proba.keys()), "Probability": list(proba.values())}
                    ).sort_values("Probability", ascending=False)
                    st.dataframe(proba_df.style.format({"Probability": "{:.1%}"}),
                                 hide_index=True, use_container_width=True)

                st.markdown('<div class="section-heading">Price &amp; Indicators</div>',
                             unsafe_allow_html=True)
                fig = make_subplots(
                    rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.2, 0.25],
                    vertical_spacing=0.04,
                    subplot_titles=(f"{ticker}", "MACD", "RSI"),
                )
                fig.add_trace(go.Candlestick(
                    x=featurized.index, open=featurized["Open"], high=featurized["High"],
                    low=featurized["Low"], close=featurized["Close"], name="Price",
                    increasing_line_color=BUY_COLOR, decreasing_line_color=SELL_COLOR,
                ), row=1, col=1)
                fig.add_trace(go.Scatter(x=featurized.index, y=featurized["sma_10"],
                                          name="SMA 10", line=dict(width=1, color=ACCENT)), row=1, col=1)
                fig.add_trace(go.Scatter(x=featurized.index, y=featurized["sma_50"],
                                          name="SMA 50", line=dict(width=1, color="#9d7bff")), row=1, col=1)
                fig.add_trace(go.Scatter(x=featurized.index, y=featurized["bb_upper"],
                                          name="BB Upper", line=dict(width=1, dash="dot", color="#B0B8C1")), row=1, col=1)
                fig.add_trace(go.Scatter(x=featurized.index, y=featurized["bb_lower"],
                                          name="BB Lower", line=dict(width=1, dash="dot", color="#B0B8C1")), row=1, col=1)

                fig.add_trace(go.Bar(x=featurized.index, y=featurized["macd_hist"], name="MACD Hist",
                                      marker_color=ACCENT), row=2, col=1)
                fig.add_trace(go.Scatter(x=featurized.index, y=featurized["macd"],
                                          name="MACD", line=dict(width=1, color="#2B3A55")), row=2, col=1)
                fig.add_trace(go.Scatter(x=featurized.index, y=featurized["macd_signal"],
                                          name="Signal", line=dict(width=1, color="#eab308")), row=2, col=1)

                fig.add_trace(go.Scatter(x=featurized.index, y=featurized["rsi_14"],
                                          name="RSI", line=dict(width=1, color="#9d7bff")), row=3, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color=SELL_COLOR, row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color=BUY_COLOR, row=3, col=1)

                fig.update_xaxes(rangeslider_visible=False, gridcolor=BORDER)
                fig.update_yaxes(gridcolor=BORDER)
                themed_fig_layout(fig, 780)
                st.plotly_chart(fig, use_container_width=True)

                vol_fig = go.Figure(go.Bar(x=featurized.index, y=featurized["Volume"],
                                            name="Volume", marker_color="#B0B8C1"))
                themed_fig_layout(vol_fig, 220)
                vol_fig.update_xaxes(gridcolor=BORDER)
                vol_fig.update_yaxes(gridcolor=BORDER)
                st.plotly_chart(vol_fig, use_container_width=True)
            except Exception as exc:
                st.error(
                    f"Something went wrong analyzing **{ticker}**: {exc}\n\n"
                    "Try a different ticker or a shorter history window."
                )

# ============================================================== PAGE: WATCHLIST
elif page == "Watchlist Suggestions":
    st.markdown('<div class="section-heading">Watchlist Scanner</div>', unsafe_allow_html=True)
    custom = st.text_input(
        "Tickers", value="", placeholder="Comma-separated, e.g. AAPL, TSLA, TCS.NS "
        "(leave blank for default watchlist)", label_visibility="collapsed"
    )
    tickers_to_scan = (
        [t.strip().upper() for t in custom.split(",") if t.strip()]
        if custom.strip() else WATCHLIST
    )

    run = st.button("Run Scan", type="primary")

    if run:
        rows = []
        progress = st.progress(0.0)
        for i, tkr in enumerate(tickers_to_scan):
            try:
                hist = fetch_history(tkr, "1y")
            except Exception:
                hist = pd.DataFrame()
            if not hist.empty and len(hist) >= 2:
                label, proba = predict_signal(hist, model, scaler, feature_columns)
                if label:
                    last_close = float(hist["Close"].iloc[-1])
                    day_change = (last_close / float(hist["Close"].iloc[-2]) - 1) * 100
                    rows.append({
                        "Ticker": tkr,
                        "Last Close": round(last_close, 2),
                        "Day Change %": round(day_change, 2),
                        "Signal": label,
                        "Confidence": round(max(proba.values()) * 100, 1),
                    })
            progress.progress((i + 1) / len(tickers_to_scan))
        progress.empty()

        if rows:
            result_df = pd.DataFrame(rows).sort_values(
                by=["Signal", "Confidence"],
                key=lambda col: col.map({"Buy": 0, "Hold": 1, "Sell": 2}) if col.name == "Signal" else col,
                ascending=[True, False],
            )

            buy_ct = (result_df["Signal"] == "Buy").sum()
            hold_ct = (result_df["Signal"] == "Hold").sum()
            sell_ct = (result_df["Signal"] == "Sell").sum()
            b1, b2, b3 = st.columns(3)
            with b1:
                stat_card("Buy-rated", str(buy_ct))
            with b2:
                stat_card("Hold-rated", str(hold_ct))
            with b3:
                stat_card("Sell-rated", str(sell_ct))

            def highlight_signal(val):
                colors = {"Buy": f"background-color: rgba(34,197,94,0.15); color:{BUY_COLOR}",
                          "Hold": f"background-color: rgba(234,179,8,0.15); color:{HOLD_COLOR}",
                          "Sell": f"background-color: rgba(239,68,68,0.15); color:{SELL_COLOR}"}
                return colors.get(val, "")

            st.markdown('<div class="section-heading">Ranked Results</div>', unsafe_allow_html=True)
            st.dataframe(
                result_df.style.map(highlight_signal, subset=["Signal"])
                .format({"Confidence": "{:.1f}%", "Day Change %": "{:+.2f}%"}),
                hide_index=True, use_container_width=True,
            )
        else:
            st.warning("Couldn't fetch data for any of the given tickers.")
    else:
        st.markdown(
            f'<div class="card">Default watchlist: '
            f'<span style="font-family:\'JetBrains Mono\',monospace;color:{ACCENT}">'
            f'{", ".join(WATCHLIST)}</span><br><br>'
            f'Click <b>Run Scan</b> to rate every stock on the list.</div>',
            unsafe_allow_html=True,
        )

# ============================================================== PAGE: INSIGHTS
elif page == "Model Insights":
    st.markdown('<div class="section-heading">Held-out Test Performance</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        stat_card("Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
    with c2:
        stat_card("Test Rows", str(metrics["n_test_rows"]))

    st.markdown('<div class="section-heading">Classification Report</div>', unsafe_allow_html=True)
    st.code(metrics["classification_report"])

    st.markdown('<div class="section-heading">Confusion Matrix</div>', unsafe_allow_html=True)
    cm_df = pd.DataFrame(metrics["confusion_matrix"],
                          index=["Actual Sell", "Actual Hold", "Actual Buy"],
                          columns=["Pred Sell", "Pred Hold", "Pred Buy"])
    st.dataframe(cm_df, use_container_width=True)

    st.markdown('<div class="section-heading">Feature Importance</div>', unsafe_allow_html=True)
    fi = pd.Series(metrics["feature_importances"]).sort_values(ascending=True)
    fi_fig = go.Figure(go.Bar(x=fi.values, y=fi.index, orientation="h", marker_color=ACCENT))
    fi_fig.update_xaxes(gridcolor=BORDER)
    fi_fig.update_yaxes(gridcolor=BORDER)
    themed_fig_layout(fi_fig, 450)
    st.plotly_chart(fi_fig, use_container_width=True)

st.markdown(
    '<div class="disclaimer">⚠️ SignalWise is a machine-learning class project '
    'trained on historical price patterns. It is not investment advice — always '
    'do your own research.</div>',
    unsafe_allow_html=True,
)
