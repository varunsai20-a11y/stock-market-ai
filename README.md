# 📈 Stock Market Prediction Using AI: Enhancing Trading Strategies

An AI-powered stock market prediction and trading strategy evaluation system built using Python, Machine Learning, Sentiment Analysis, and Streamlit.

---

## 🚀 Project Overview

This project predicts stock prices using machine learning, combines stock-related news sentiment, and generates Buy / Sell / Hold trading signals. It also includes backtesting to compare the AI strategy against traditional strategies.

---

## 🎯 Features

- Predict next-day stock price
- Analyze recent stock-related news sentiment
- Generate Buy / Sell / Hold recommendations
- Compare AI strategy against:
  - Buy & Hold
  - Moving Average Crossover
- Interactive Streamlit dashboard
- Evaluation metrics:
  - RMSE
  - MAE
  - Direction Accuracy
  - Sharpe Ratio
  - Max Drawdown

---

## 🛠️ Tech Stack

- Python
- Pixi
- pandas
- numpy
- scikit-learn
- yfinance
- TextBlob
- Streamlit
- Plotly

---

## 📂 Project Structure

```bash
stock_market_ai/
│
├── app.py
├── main.py
├── model.py
├── sentiment.py
├── strategy.py
├── backtest.py
├── price_feed.py
├── utils.py
├── pixi.toml
├── README.md
│
├── data/
├── models/
├── outputs/
└── screenshots/