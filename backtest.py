import pandas as pd
import numpy as np

from models import add_features, ALL_FEATURE_COLS
from lstm_model import train_lstm_model, SEQ_LEN
import joblib
from strategy import decide_trade, execute_trade
from sentiment import fetch_sentiment
from price_feed import fetch_stock_data
from utils import sharpe_ratio, max_drawdown, ensure_directories


def run_ai_backtest(ticker="AAPL", initial_cash=10000):
    ensure_directories()
    df_raw = fetch_stock_data(ticker, start="2022-01-01", end="2024-12-31")
    df = add_features(df_raw)

    model, full_df, metrics, _, _ = train_lstm_model(ticker=ticker, start="2020-01-01", end="2024-12-31")

    scaler_X = joblib.load(f"models/{ticker}_scaler_X.pkl")

    cash = initial_cash
    holdings = 0
    portfolio_values = []
    trade_log = []

    for i in range(SEQ_LEN, len(df) - 1):
        current_price = df.iloc[i]["Close"]
        current_date = df.index[i]

        sequence = df.iloc[i-SEQ_LEN+1:i+1][ALL_FEATURE_COLS].values
        sequence_scaled = scaler_X.transform(sequence)
        
        predicted_out = model.predict(np.array([sequence_scaled]), verbose=0)
        predicted_probs = predicted_out[0][0]  # Multi-task model: [class_output, price_output]
        predicted_class = int(np.argmax(predicted_probs))
        confidence = float(predicted_probs[predicted_class])
        
        action_map = {0: "Sell", 1: "Hold", 2: "Buy"}
        predicted_action = action_map[predicted_class]

        # Zero out sentiment for historical backtests to prevent data leakage/hallucination
        sentiment_score = 0.0
        action, reason = decide_trade(sentiment_score, predicted_action, confidence)

        cash, holdings = execute_trade(action, cash, holdings, current_price)

        portfolio_value = cash + holdings * current_price
        portfolio_values.append(portfolio_value)

        trade_log.append({
            "Date": current_date,
            "Price": current_price,
            "Predicted Signal": predicted_action,
            "Confidence": confidence,
            "Sentiment": sentiment_score,
            "Action": action,
            "Portfolio Value": portfolio_value,
            "Reason": reason
        })

    log_df = pd.DataFrame(trade_log)
    log_df.to_csv(f"outputs/{ticker}_ai_backtest.csv", index=False)

    returns = log_df["Portfolio Value"].pct_change().dropna()

    total_return = ((log_df["Portfolio Value"].iloc[-1] - initial_cash) / initial_cash) * 100
    sharpe = sharpe_ratio(returns)
    drawdown = max_drawdown(log_df["Portfolio Value"].values)

    results = {
        "Strategy": "AI Strategy",
        "Final Portfolio Value": round(log_df["Portfolio Value"].iloc[-1], 2),
        "Total Return (%)": round(total_return, 2),
        "Sharpe Ratio": sharpe,
        "Max Drawdown (%)": drawdown
    }

    return log_df, results


def run_buy_and_hold_backtest(ticker="AAPL", initial_cash=10000):
    df = fetch_stock_data(ticker, start="2022-01-01", end="2024-12-31")

    buy_price = df["Close"].iloc[0]
    shares = (initial_cash * 0.999) / buy_price
    portfolio_values = shares * df["Close"]

    returns = portfolio_values.pct_change().dropna()

    total_return = ((portfolio_values.iloc[-1] - initial_cash) / initial_cash) * 100
    sharpe = sharpe_ratio(returns)
    drawdown = max_drawdown(portfolio_values.values)

    results = {
        "Strategy": "Buy & Hold",
        "Final Portfolio Value": round(portfolio_values.iloc[-1], 2),
        "Total Return (%)": round(total_return, 2),
        "Sharpe Ratio": sharpe,
        "Max Drawdown (%)": drawdown
    }

    return results


def run_ma_crossover_backtest(ticker="AAPL", initial_cash=10000):
    df = fetch_stock_data(ticker, start="2022-01-01", end="2024-12-31").copy()

    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_30"] = df["Close"].rolling(30).mean()
    df.dropna(inplace=True)

    cash = initial_cash
    holdings = 0
    portfolio_values = []

    for _, row in df.iterrows():
        price = row["Close"]

        if row["SMA_10"] > row["SMA_30"] and cash > 0:
            holdings = (cash * 0.999) / price
            cash = 0

        elif row["SMA_10"] < row["SMA_30"] and holdings > 0:
            cash = holdings * price * 0.999
            holdings = 0

        portfolio_values.append(cash + holdings * price)

    portfolio_values = pd.Series(portfolio_values, index=df.index)
    returns = portfolio_values.pct_change().dropna()

    total_return = ((portfolio_values.iloc[-1] - initial_cash) / initial_cash) * 100
    sharpe = sharpe_ratio(returns)
    drawdown = max_drawdown(portfolio_values.values)

    results = {
        "Strategy": "MA Crossover",
        "Final Portfolio Value": round(portfolio_values.iloc[-1], 2),
        "Total Return (%)": round(total_return, 2),
        "Sharpe Ratio": sharpe,
        "Max Drawdown (%)": drawdown
    }

    return results


if __name__ == "__main__":
    ticker = "AAPL"

    ai_log, ai_results = run_ai_backtest(ticker)
    buy_hold_results = run_buy_and_hold_backtest(ticker)
    ma_results = run_ma_crossover_backtest(ticker)

    print("\nBacktest Results:\n")
    print(ai_results)
    print(buy_hold_results)
    print(ma_results)