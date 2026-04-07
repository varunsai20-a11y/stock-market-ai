import os
import joblib
import numpy as np
import pandas as pd

from price_feed import fetch_stock_data
from utils import ensure_directories

FEATURES = ["Open", "High", "Low", "Close", "Volume"]
ALL_FEATURE_COLS = FEATURES + ["SMA_5", "SMA_10", "Return_1d", "Volatility_5", "RSI_14", "MACD", "MACD_Signal", "BB_Upper", "BB_Lower", "ATR_14", "ROC_5"]


def add_features(df):
    """
    Add technical features for prediction.
    """
    df = df.copy()

    df["SMA_5"] = df["Close"].rolling(5).mean()
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["Return_1d"] = df["Close"].pct_change()
    df["Volatility_5"] = df["Close"].rolling(5).std()

    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    sma_20 = df["Close"].rolling(20).mean()
    std_20 = df["Close"].rolling(20).std()
    df["BB_Upper"] = sma_20 + (std_20 * 2)
    df["BB_Lower"] = sma_20 - (std_20 * 2)
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR_14'] = true_range.rolling(14).mean()

    # Momentum (Rate of Change) - 5 days
    df['ROC_5'] = df['Close'].pct_change(periods=5) * 100

    # Categorical target based on 1-day future return
    future_return = df["Close"].shift(-1) / df["Close"] - 1
    
    # 0 = Sell, 1 = Hold, 2 = Buy
    threshold = 0.005 # 0.5%
    conditions = [
        (future_return > threshold),
        (future_return < -threshold)
    ]
    choices = [2, 0] # Buy, Sell
    # Using np.select returns an array, we assign it as floats so we can keep NaN
    targets = np.select(conditions, choices, default=1).astype(float)
    
    df["Target"] = targets
    df.loc[df["Close"].shift(-1).isna(), "Target"] = np.nan

    # Drop only rows where features are NaN, keep row with Target=NaN (today)
    df.dropna(subset=ALL_FEATURE_COLS, inplace=True)
    return df


