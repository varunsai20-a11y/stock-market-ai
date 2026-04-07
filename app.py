import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from textblob import TextBlob

from lstm_model import train_lstm_model, forecast_next_price_lstm
from sentiment import fetch_sentiment
from strategy import decide_trade
from backtest import run_ai_backtest, run_buy_and_hold_backtest, run_ma_crossover_backtest
from price_feed import fetch_stock_data


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Market AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Stock Market AI")
    st.caption("LSTM · Sentiment · Backtesting")
    st.divider()

    ticker = st.text_input("Stock ticker", "AAPL", max_chars=10).upper().strip()

    run_button = st.button("Run analysis", type="primary", use_container_width=True)

    st.divider()
    if st.button("Clear cache", use_container_width=True):
        st.cache_resource.clear()
        st.session_state.clear()
        st.success("Cache cleared. Re-run analysis to reload.")

    st.caption(
        "Cache is per ticker. Switch tickers freely — "
        "use **Clear cache** only if you want to force a full re-train."
    )


# ── Cached model training ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_trained_model(ticker_name: str):
    return train_lstm_model(ticker=ticker_name)


# ── Helper: colour a recommendation ──────────────────────────────────────────
def show_recommendation(action: str, reason: str):
    if action == "Buy":
        st.success(f"**Recommendation: {action}** — {reason}", icon="🟢")
    elif action == "Sell":
        st.error(f"**Recommendation: {action}** — {reason}", icon="🔴")
    else:
        st.warning(f"**Recommendation: {action}** — {reason}", icon="🟡")


# ── Helper: style backtest table ──────────────────────────────────────────────
def style_backtest(df: pd.DataFrame):
    def colour_return(val):
        try:
            v = float(val)
            return "color: #2e7d32" if v > 0 else "color: #c62828"
        except (ValueError, TypeError):
            return ""

    def colour_drawdown(val):
        try:
            v = float(val)
            return "color: #c62828" if v < 0 else ""
        except (ValueError, TypeError):
            return ""

    return (
        df.style
        .applymap(colour_return, subset=["Total Return (%)"])
        .applymap(colour_drawdown, subset=["Max Drawdown (%)"])
        .format({"Final Portfolio Value": "${:,.2f}", "Total Return (%)": "{:.2f}%",
                 "Sharpe Ratio": "{:.2f}", "Max Drawdown (%)": "{:.2f}%"})
        .set_properties(**{"text-align": "right"}, subset=pd.IndexSlice[:, df.columns[1:]])
        .set_properties(**{"font-weight": "bold"}, subset=pd.IndexSlice[:, ["Strategy"]])
    )


# ── Helper: sentiment progress bar ───────────────────────────────────────────
def show_sentiment_bar(score: float):
    # Map [-1, 1] → [0, 100]
    progress_val = int((score + 1) * 50)
    label = (
        "Very positive" if score > 0.5 else
        "Positive"      if score > 0.1 else
        "Neutral"       if score > -0.1 else
        "Negative"      if score > -0.5 else
        "Very negative"
    )
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.progress(progress_val, text=f"Sentiment: **{label}**")
    with col_b:
        st.metric("Score", f"{score:+.3f}", label_visibility="collapsed")


# ── Main content ──────────────────────────────────────────────────────────────
st.title(f"Stock analysis · {ticker}")

if run_button:
    st.session_state["run_analysis"] = True
    st.session_state["current_ticker"] = ticker
    st.session_state["backtest_done"] = False

if not st.session_state.get("run_analysis") or st.session_state.get("current_ticker") != ticker:
    st.info("Enter a ticker in the sidebar and press **Run analysis** to begin.", icon="ℹ️")
    st.stop()

# ── Phase 1: fetch & train ────────────────────────────────────────────────────
with st.status("Running analysis…", expanded=True) as status:
    try:
        st.write("📥 Fetching historical price data…")
        # Trigger a lightweight pre-fetch so the user sees progress before training
        _ = fetch_stock_data(ticker, start="2020-01-01", end="2024-12-31")

        st.write("🧠 Training LSTM model (this takes ~30–60 s the first time)…")
        model, df, metrics, actual, predicted = get_trained_model(ticker)

        st.write("🔮 Generating forecast & sentiment…")
        predicted_action, confidence, predicted_probs = forecast_next_price_lstm(model, df)
        current_price = df["Close"].iloc[-1]
        sentiment_score, headlines = fetch_sentiment(ticker)
        action, reason = decide_trade(sentiment_score, predicted_action, confidence)

        status.update(label="Analysis complete!", state="complete", expanded=False)
    except Exception as e:
        status.update(label="Analysis failed", state="error", expanded=True)
        st.error(f"Data Fetch Error: {e}")
        st.info("Since fallback synthetic data was removed, yfinance API hits will halt the app if limited. Try again later.")
        st.stop()


# ── Overview metrics ──────────────────────────────────────────────────────────
st.subheader("Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current price",    f"${current_price:,.2f}")
col2.metric("Predicted signal", predicted_action)
col3.metric("Model confidence", f"{confidence*100:.1f}%")
col4.metric("Sentiment score",  f"{sentiment_score:+.3f}")

st.divider()


# ── Recommendation ────────────────────────────────────────────────────────────
show_recommendation(action, reason)

st.divider()


# ── Two-column: model metrics + sentiment/news ────────────────────────────────
left, right = st.columns(2, gap="large")

with left:
    st.subheader("Model evaluation")
    metrics_df = pd.DataFrame([metrics]).T.rename(columns={0: "Value"})
    st.dataframe(metrics_df, use_container_width=True)

    st.caption(
        "A ~52% test direction accuracy on real AAPL data is normal — "
        "liquid equities approach a random walk. Train accuracy reflects in-sample fit."
    )

with right:
    st.subheader("Sentiment")
    show_sentiment_bar(sentiment_score)

    st.markdown("**Recent headlines**")
    if headlines:
        for h in headlines[:6]:
            polarity = round(TextBlob(h).sentiment.polarity, 2)
            badge = "🟢" if polarity > 0.05 else "🔴" if polarity < -0.05 else "⚪"
            st.markdown(f"{badge} {h}  `{polarity:+.2f}`")
    else:
        st.caption("No recent headlines found.")

st.divider()


# ── Class Probabilities chart ─────────────────────────────────────────────────
st.subheader("Current prediction probabilities")

fig_pred = go.Figure(data=[
    go.Bar(name='Probabilities', x=['Sell', 'Hold', 'Buy'], y=predicted_probs, marker_color=['#E53935', '#FB8C00', '#43A047'])
])
fig_pred.update_layout(
    yaxis_title="Probability",
    margin=dict(l=0, r=0, t=30, b=0),
    height=320,
)
st.plotly_chart(fig_pred, use_container_width=True)

st.divider()


# ── Backtesting ───────────────────────────────────────────────────────────────
st.subheader("Backtesting")
st.caption("2022-01-01 → 2024-12-31 · $10,000 starting capital · 0.1% transaction cost")

run_backtest = st.button("Run backtesting", use_container_width=False)

if run_backtest or st.session_state.get("backtest_done"):
    try:
        if run_backtest:
            with st.spinner("Running three strategies…"):
                ai_log, ai_results         = run_ai_backtest(ticker)
                buy_hold_results           = run_buy_and_hold_backtest(ticker)
                ma_results                 = run_ma_crossover_backtest(ticker)

                st.session_state["backtest_done"]    = True
                st.session_state["ai_log"]           = ai_log
                st.session_state["ai_results"]       = ai_results
                st.session_state["buy_hold_results"] = buy_hold_results
                st.session_state["ma_results"]       = ma_results

        # Load from session state if already computed
        ai_log           = st.session_state["ai_log"]
        ai_results       = st.session_state["ai_results"]
        buy_hold_results = st.session_state["buy_hold_results"]
        ma_results       = st.session_state["ma_results"]

        comparison_df = pd.DataFrame([ai_results, buy_hold_results, ma_results])
        st.dataframe(style_backtest(comparison_df), use_container_width=True, hide_index=True)

        # ── Combined portfolio growth chart ───────────────────────────────────────
        st.markdown("**Portfolio growth — all strategies**")

        # Reconstruct buy & hold + MA series for charting
        df_bh = fetch_stock_data(ticker, start="2022-01-01", end="2024-12-31")
        buy_price = df_bh["Close"].iloc[0]
        bh_series = (10000 * 0.999 / buy_price) * df_bh["Close"]

        df_ma = df_bh.copy()
        df_ma["SMA_10"] = df_ma["Close"].rolling(10).mean()
        df_ma["SMA_30"] = df_ma["Close"].rolling(30).mean()
        df_ma.dropna(inplace=True)
        cash_ma, hold_ma, ma_vals = 10000, 0, []
        for _, row in df_ma.iterrows():
            p = row["Close"]
            if row["SMA_10"] > row["SMA_30"] and cash_ma > 0:
                hold_ma = (cash_ma * 0.999) / p; cash_ma = 0
            elif row["SMA_10"] < row["SMA_30"] and hold_ma > 0:
                cash_ma = hold_ma * p * 0.999; hold_ma = 0
            ma_vals.append(cash_ma + hold_ma * p)
        ma_series = pd.Series(ma_vals, index=df_ma.index)

        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(
            x=ai_log["Date"], y=ai_log["Portfolio Value"],
            mode="lines", name="AI strategy",
            line=dict(color="#7B1FA2", width=2)
        ))
        fig_bt.add_trace(go.Scatter(
            x=bh_series.index, y=bh_series.values,
            mode="lines", name="Buy & hold",
            line=dict(color="#1565C0", width=2, dash="dot")
        ))
        fig_bt.add_trace(go.Scatter(
            x=ma_series.index, y=ma_series.values,
            mode="lines", name="MA crossover",
            line=dict(color="#F57C00", width=2, dash="dash")
        ))
        fig_bt.add_hline(
            y=10000, line_dash="dot", line_color="gray",
            annotation_text="Starting capital", annotation_position="bottom right"
        )
        fig_bt.update_layout(
            xaxis_title="Date",
            yaxis_title="Portfolio value (USD)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=30, b=0),
            height=360,
        )
        st.plotly_chart(fig_bt, use_container_width=True)

        # ── AI trade log expander ─────────────────────────────────────────────────
        with st.expander("View AI trade log"):
            st.dataframe(
                ai_log[["Date", "Price", "Predicted Signal", "Confidence", "Sentiment", "Action", "Portfolio Value"]]
                .style.format({
                    "Price": "${:.2f}", "Confidence": "{:.2f}",
                    "Sentiment": "{:.3f}", "Portfolio Value": "${:,.2f}"
                }),
                use_container_width=True,
                height=320,
            )
    except Exception as e:
        st.error(f"Backtesting failed: {e}")

else:
    st.info("Press **Run backtesting** to compare strategies. "
            "This runs separately from training to avoid re-training the model.", icon="ℹ️")