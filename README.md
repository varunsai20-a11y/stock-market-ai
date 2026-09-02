# 📈 Stock Market Prediction Using AI: Enhancing Trading Strategies

An end-to-end machine learning and deep learning framework for algorithmic trading and stock trend forecasting. The system integrates technical indicators, time-series forecasting via Multi-Task LSTM networks, ensemble classification with XGBoost, and real-time financial news sentiment analysis served through an interactive Streamlit dashboard.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Pipeline](#-system-pipeline)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
- [Model Architecture & Methodology](#-model-architecture--methodology)
- [Screenshots & Dashboard](#-screenshots--dashboard)
- [Author](#-author)
- [License](#-license)

---

## 📖 Overview

Predicting stock market movements requires capturing both historical price dynamics and real-time market sentiment. This project combines:
1. **Quantitative Market Data:** Historical OHLCV metrics, momentum oscillators (RSI, MACD), and moving averages.
2. **Qualitative Sentiment Signals:** Financial news headlines processed via NLP to generate sentiment polarity scores.
3. **Hybrid Modeling:** Deep recurrent networks (LSTM) for continuous price trajectory regression paired with tree-based ensembles (XGBoost) for buy/sell/hold signal classification.

---

## ✨ Key Features

- **📊 Multi-Task Time-Series Forecasting:** Multi-layer LSTM architecture predicting directional trends and future closing prices.
- **🌲 Decision Classification:** XGBoost classifier delivering actionable trade signals (Buy / Sell / Hold) with probability confidences.
- **📰 Financial News Sentiment Analysis:** Real-time sentiment scoring using NLP pipelines to factor macroeconomic events into trading signals.
- **🖥️ Streamlit Interactive UI:** Dynamic dashboard allowing users to input ticker symbols, select prediction horizons, view technical overlays, and backtest strategies.
- **📈 Backtesting & Performance Metrics:** Evaluates accuracy, RMSE, MAE, Sharpe Ratio, and cumulative returns against buy-and-hold baselines.

---

## 🏗️ System Pipeline

```text
+-----------------------+     +------------------------+
|  Historical Data API  |     | Financial News Feed    |
|  (Yahoo Finance/Alpha)|     | (RSS / News API)       |
+-----------+-----------+     +-----------+------------+
            |                             |
            v                             v
+-----------------------+     +------------------------+
| Technical Indicators  |     | NLP Sentiment Pipeline |
| (RSI, MACD, SMA, EMA) |     | (VADER / FinBERT)      |
+-----------+-----------+     +-----------+------------+
            |                             |
            +------------+  +-------------+
                         |  |
                         v  v
            +-----------------------------+
            |   Feature Engineering &     |
            |     Scaling Pipeline        |
            +--------------+--------------+
                           |
             +-------------+-------------+
             |                           |
             v                           v
+-------------------------+ +-------------------------+
|  Multi-Task LSTM Model  | |   XGBoost Classifier    |
| (Price & Trend Horizon) | | (Trade Signal Generation|
+------------+------------+ +------------+------------+
             |                           |
             +-------------+-------------+
                           |
                           v
            +-----------------------------+
            |     Streamlit Dashboard     |
            |  (Interactive Visualizer)   |
            +-----------------------------+
```

---

## 🛠️ Tech Stack

| Component | Technologies |
|---|---|
| **Language** | Python 3.10+ |
| **Deep Learning & ML** | PyTorch / TensorFlow, Scikit-Learn, XGBoost |
| **NLP & Sentiment** | NLTK, VADER, Hugging Face Transformers |
| **Data Processing & Finance** | Pandas, NumPy, yfinance, TA-Lib |
| **Visualization & App UI** | Streamlit, Plotly, Matplotlib, Seaborn |

---

## 📂 Project Structure

```text
stock-market-ai/
├── data/                   # Cached raw data and processed datasets
├── models/                 # Saved model weights (.pkl, .pt, or .h5)
├── notebooks/              # Exploratory data analysis & model experiments
├── src/
│   ├── data_loader.py      # Historical ticker data fetching & cleaning
│   ├── feature_eng.py      # Technical indicator calculation & scaling
│   ├── sentiment.py        # Financial news scraping & NLP sentiment scoring
│   ├── train_lstm.py       # LSTM model training & checkpointing
│   ├── train_xgboost.py    # XGBoost classifier training
│   └── evaluate.py         # Backtesting engine & metrics evaluation
├── app.py                  # Main Streamlit dashboard application
├── requirements.txt        # Python package dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Getting Started

### Prerequisites
- Python `3.10` or higher
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/varunsai20-a11y/stock-market-ai.git](https://github.com/varunsai20-a11y/stock-market-ai.git)
   cd stock-market-ai
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # macOS / Linux
   python3 -m
