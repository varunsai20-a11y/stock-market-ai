import os
import joblib
import numpy as np
import pandas as pd

from price_feed import fetch_stock_data
from utils import ensure_directories

FEATURES = ["Open", "High", "Low", "Close", "Volume"]
ALL_FEATURE_COLS = FEATURES + [
    "SMA_5", "SMA_10", "Return_1d", "Volatility_5", "RSI_14", "MACD", "MACD_Signal", 
    "BB_Upper", "BB_Lower", "ATR_14", "ROC_5", "EMA_20", "SMA_50", "Volume_Change_Pct",
    "OBV", "CCI_14", "Stoch_K", "Stoch_D", "MFI_14", "Force_Index", "Dist_SMA_10", "Dist_SMA_50"
]


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

    # New requested indicators
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['Volume_Change_Pct'] = df['Volume'].pct_change() * 100

    # Advanced Indicators
    # OBV
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # CCI
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    tp_sma = tp.rolling(14).mean()
    tp_mad = tp.rolling(14).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df['CCI_14'] = (tp - tp_sma) / (0.015 * tp_mad)

    # Stochastic Oscillator
    low_14 = df['Low'].rolling(14).min()
    high_14 = df['High'].rolling(14).max()
    df['Stoch_K'] = (df['Close'] - low_14) / (high_14 - low_14) * 100
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    # MFI
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    pos_mf = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
    neg_mf = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
    mfr = pos_mf / neg_mf
    df['MFI_14'] = 100 - (100 / (1 + mfr))

    # Force Index
    df['Force_Index'] = df['Close'].diff(1) * df['Volume']

    # Relative Distance
    df['Dist_SMA_10'] = (df['Close'] - df['SMA_10']) / df['SMA_10']
    df['Dist_SMA_50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']

    # Binary target: 1 = Up (price rises > threshold), 0 = Down
    future_return = df["Close"].shift(-1) / df["Close"] - 1
    threshold = 0.005  # 0.5% move required

    df["Target"] = (future_return > threshold).astype(float)
    df.loc[df["Close"].shift(-1).isna(), "Target"] = np.nan

    # Generate 7-day forward predicted prices
    for i in range(1, 8):
        df[f"Target_Price_{i}"] = df["Close"].shift(-i)

    # Drop only rows where features are NaN. 
    # Do NOT drop NaN target rows because we need the latest row for inference!
    df.dropna(subset=ALL_FEATURE_COLS, inplace=True)
    return df


