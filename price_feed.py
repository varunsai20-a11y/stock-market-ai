import yfinance as yf
import pandas as pd
import requests
import numpy as np
import pandas_datareader.data as web
import time
import os
from pathlib import Path

session = requests.Session()
session.headers.update(
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
)

CACHE_DIR = Path(__file__).parent / "data"
CACHE_DIR.mkdir(exist_ok=True)


def get_cache_path(ticker):
    """Get the cache file path for a ticker."""
    return CACHE_DIR / f"{ticker}_cached.csv"


def load_from_cache(ticker):
    """Load cached data if available."""
    cache_path = get_cache_path(ticker)
    if cache_path.exists():
        try:
            df = pd.read_csv(cache_path, index_col='Date', parse_dates=True)
            if not df.empty and 'Close' in df.columns:
                print(f"[✓] Loaded cached data for {ticker}")
                return df
        except Exception as e:
            print(f"[!] Failed to load cache for {ticker}: {e}")
    return None


def save_to_cache(ticker, df):
    """Save data to cache file."""
    try:
        cache_path = get_cache_path(ticker)
        df.to_csv(cache_path)
        print(f"[✓] Cached data for {ticker}")
    except Exception as e:
        print(f"[!] Failed to cache data for {ticker}: {e}")


def fetch_stock_data(ticker="AAPL", start="2020-01-01", end="2024-12-31", max_retries=3):
    """
    Fetch historical stock data using yfinance with retry logic. 
    Falls back to stooq, then cached data if available.
    """
    for attempt in range(max_retries):
        try:
            print(f"[Attempt {attempt + 1}] Fetching {ticker} from yfinance...")
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, session=session, multi_level_index=False, progress=False)

            if df.empty:
                raise ValueError(f"No data found for ticker: {ticker}")

            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df.dropna(inplace=True)
            save_to_cache(ticker, df)
            return df
        except Exception as e_yf:
            print(f"[!] yfinance error (attempt {attempt + 1}/{max_retries}): {e_yf}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"   Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
    
    # Try stooq as fallback
    try:
        print(f"[!] yfinance failed, trying stooq fallback...")
        df = web.DataReader(ticker, 'stooq', start=start, end=end)
        if df.empty:
            raise ValueError(f"No data found for ticker {ticker} on stooq.")
        
        # Stooq returns descending index, sort it
        df = df.sort_index()
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.dropna(inplace=True)
        save_to_cache(ticker, df)
        return df
    except Exception as e_stooq:
        print(f"[!] Stooq fallback failed: {e_stooq}")
    
    # Try cached data as last resort
    cached_df = load_from_cache(ticker)
    if cached_df is not None:
        print(f"[!] All live sources failed. Using cached data (may be outdated).")
        return cached_df
    
    raise ValueError(f"All data sources failed for {ticker}. Please try again later or check your internet connection.")


def get_live_price(ticker="AAPL", max_retries=2):
    """
    Fetch the latest closing price with retry logic.
    """
    for attempt in range(max_retries):
        try:
            print(f"[Attempt {attempt + 1}] Fetching live price for {ticker}...")
            stock = yf.Ticker(ticker, session=session)
            hist = stock.history(period="1d", progress=False)

            if hist.empty:
                raise ValueError(f"Could not fetch live price for {ticker}")

            price = float(hist["Close"].iloc[-1])
            print(f"[✓] Got live price: ${price:.2f}")
            return price
        except Exception as e:
            print(f"[!] Error fetching live price (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"   Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
    
    raise ValueError(f"Could not fetch live price for {ticker} after {max_retries} attempts. Please try again later.")


def fetch_news_headlines(ticker="AAPL", max_items=10):
    """
    Fetch recent stock-related news headlines using yfinance.
    Returns empty list if unable to fetch.
    """
    try:
        stock = yf.Ticker(ticker, session=session)
        news = stock.news

        headlines = []

        if news:
            for item in news[:max_items]:
                title = item.get("title", "")
                if title:
                    headlines.append(title)

        return headlines
    except Exception as e:
        print(f"[!] Warning: Could not fetch news for {ticker}: {e}")
        return []  # Return empty list instead of crashing