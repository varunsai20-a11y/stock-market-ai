import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
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

@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_data(ticker_name: str, start: str, end: str):
    return fetch_stock_data(ticker_name, start=start, end=end)


# ── Helper: colour a recommendation ──────────────────────────────────────────
def show_recommendation(action: str, reason: str):
    colors = {"Buy": "#4CAF50", "Sell": "#F44336", "Hold": "#FF9800"}
    color = colors.get(action, "gray")
    st.markdown(f"""
        <div style="border-left: 5px solid {color}; padding: 15px; border-radius: 5px; background-color: rgba(255,255,255,0.05); margin-bottom: 20px;">
            <p style="margin: 0; font-size: 18px; font-weight: bold; color: {color};">Recommendation: {action}</p>
            <p style="margin: 5px 0 0 0; color: #e0e0e0; font-size: 15px;">{reason}</p>
        </div>
    """, unsafe_allow_html=True)


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
        .map(colour_return, subset=["Total Return (%)"])
        .map(colour_drawdown, subset=["Max Drawdown (%)"])
        .format({"Final Portfolio Value": "${:,.2f}", "Total Return (%)": "{:.2f}%",
                 "Sharpe Ratio": "{:.2f}", "Max Drawdown (%)": "{:.2f}%"})
        .set_properties(**{"text-align": "right"}, subset=pd.IndexSlice[:, df.columns[1:]])
        .set_properties(**{"font-weight": "bold"}, subset=pd.IndexSlice[:, ["Strategy"]])
    )


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
        _ = get_stock_data(ticker, start="2020-01-01", end="2024-12-31")

        st.write("🧠 Training Dual-Head LSTM model (this takes ~30–60 s the first time)…")
        model, df, metrics, actual, predicted = get_trained_model(ticker)

        st.write("🔮 Generating forecast & sentiment…")
        predicted_action, confidence, predicted_probs, predicted_price_seq = forecast_next_price_lstm(model, df, ticker)
        
        current_price = df["Close"].iloc[-1]
        predicted_price_1d = predicted_price_seq[0]
        price_delta_1d = predicted_price_1d - current_price
        price_delta_pct = (price_delta_1d / current_price) * 100
        
        sentiment_score, headlines = fetch_sentiment(ticker)
        action, reason = decide_trade(sentiment_score, predicted_action, confidence)

        status.update(label="Analysis complete!", state="complete", expanded=False)
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        status.update(label="Analysis failed", state="error", expanded=True)
        import traceback
        st.error(f"Error: {e}\n\n{traceback.format_exc()}")
        st.info("Since fallback synthetic data was removed, yfinance API hits will halt the app if limited. Try again later.")
        st.stop()


# ── Overview metrics ──────────────────────────────────────────────────────────
st.subheader("Overview")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Current price", f"${current_price:,.2f}")
col2.metric("Predicted signal", predicted_action)
col3.metric("Predicted Price (Next Day)", f"${predicted_price_1d:,.2f}", delta=f"{price_delta_pct:+.2f}%")

with col4:
    conf_color_hex = "#4CAF50" if confidence > 0.7 else "#FF9800" if confidence > 0.5 else "#F44336"
    st.markdown(f"<p style='font-size:14px; margin-bottom:0;'>Model confidence</p><h2 style='color:{conf_color_hex}; margin-top:0;'>{confidence*100:.1f}%</h2>", unsafe_allow_html=True)
    st.progress(confidence)

with col5:
    sent_color_hex = "#4CAF50" if sentiment_score > 0.1 else "#F44336" if sentiment_score < -0.1 else "#9E9E9E"
    st.markdown(f"<p style='font-size:14px; margin-bottom:0;'>Sentiment score</p><h2 style='color:{sent_color_hex}; margin-top:0;'>{sentiment_score:+.3f}</h2>", unsafe_allow_html=True)

st.divider()


# ── Recommendation & Stop Loss Widget ──────────────────────────────────────────
show_recommendation(action, reason)

# Stop Loss Table
st.markdown("### Trade Parameters")
target_price = max(predicted_price_seq) if max(predicted_price_seq) > current_price else current_price * 1.05
sl_price = current_price * 0.98
rr_ratio = (target_price - current_price) / (current_price - sl_price) if target_price > current_price else 0

trade_df = pd.DataFrame([{
    "Entry Price": f"${current_price:.2f}",
    "Target Price": f"${target_price:.2f}",
    "Stop Loss": f"${sl_price:.2f}",
    "Risk/Reward Ratio": f"{rr_ratio:.2f}"
}])
st.dataframe(trade_df, hide_index=True)

st.divider()


# ── Forecast Chart ────────────────────────────────────────────────────────────
st.subheader("7-Day Forecast & Historical Prices")
hist_dates = df.index[-60:]
hist_prices = df['Close'].iloc[-60:]

last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7, freq='B')

fig_forecast = go.Figure()
fig_forecast.add_trace(go.Scatter(x=hist_dates, y=hist_prices, mode='lines', name='Actual Price', line=dict(color='#4CAF50', width=2)))
fig_forecast.add_trace(go.Scatter(x=future_dates, y=predicted_price_seq, mode='lines', name='Forecasted 7D', line=dict(color='#FF9800', width=2, dash='dash')))
fig_forecast.add_vline(x=last_date, line_dash="dot", line_color="rgba(255,255,255,0.3)")

fig_forecast.update_layout(
    xaxis_title="Date", 
    yaxis_title="Price (USD)", 
    height=400,
    margin=dict(l=0, r=0, t=30, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_forecast, use_container_width=True)

st.divider()

# ── Two-column: model metrics + sentiment/news ────────────────────────────────
left, right = st.columns(2, gap="large")

with left:
    st.subheader("Model Evaluation")
    metrics_df = pd.DataFrame([metrics]).T.rename(columns={0: "Value"})
    st.dataframe(metrics_df, use_container_width=True)

    st.caption("Test classification accuracy metrics for the Sell/Hold/Buy signal target.")
    
    st.markdown("**Current prediction probabilities**")
    fig_pred = go.Figure(data=[
        go.Bar(name='Probabilities', x=['Sell', 'Hold', 'Buy'], y=predicted_probs, marker_color=['#E53935', '#FB8C00', '#43A047'])
    ])
    fig_pred.update_layout(
        yaxis_title="Probability",
        margin=dict(l=0, r=0, t=10, b=0),
        height=250,
    )
    st.plotly_chart(fig_pred, use_container_width=True)

with right:
    st.subheader("Sentiment NLP")
    
    badge = "Bullish 🟢" if sentiment_score > 0.1 else "Bearish 🔴" if sentiment_score < -0.1 else "Neutral ⚪"
    st.markdown(f"**Overall Signal:** {badge}")
    st.markdown("<br>", unsafe_allow_html=True)
    
    if headlines:
        for h in headlines[:6]:
            polarity = round(TextBlob(h).sentiment.polarity, 2)
            bg_col = "rgba(76, 175, 80, 0.1)" if polarity > 0.05 else "rgba(244, 67, 54, 0.1)" if polarity < -0.05 else "rgba(158, 158, 158, 0.1)"
            bord_col = "#4CAF50" if polarity > 0.05 else "#F44336" if polarity < -0.05 else "#9E9E9E"
            st.markdown(f"""
                <div style="border-left: 4px solid {bord_col}; padding: 12px; margin-bottom: 12px; background-color: {bg_col}; border-radius: 4px;">
                    <span style="font-size:14px; font-weight:500;">{h}</span> 
                    <span style="float:right; font-weight:bold; color:{bord_col}; padding-left:10px;">{polarity:+.2f}</span>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("No recent headlines found.")

st.divider()


# ── Backtesting ───────────────────────────────────────────────────────────────
st.subheader("Historical Verification")
st.caption("2022-01-01 → 2024-12-31 · $10,000 starting capital · 0.1% transaction cost")

run_backtest = st.button("Run backtesting engine", use_container_width=False)

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
        import traceback
        st.error(f"Backtesting failed: {e}\n\n{traceback.format_exc()}")

else:
    st.info("Press **Run backtesting engine** to compare strategies. "
            "This runs separately from training to avoid re-training the model.", icon="ℹ️")