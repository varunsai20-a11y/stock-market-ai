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
                print(f"[OK] Loaded cached data for {ticker}")
                return df
        except Exception as e:
            print(f"[!] Failed to load cache for {ticker}: {e}")
    return None


def save_to_cache(ticker, df):
    """Save data to cache file."""
    try:
        cache_path = get_cache_path(ticker)
        df.to_csv(cache_path)
        print(f"[OK] Cached data for {ticker}")
    except Exception as e:
        print(f"[!] Failed to cache data for {ticker}: {e}")


def fetch_stock_data(ticker="AAPL", start="2020-01-01", end=None, max_retries=3):
    """
    Fetch historical stock data using yfinance with retry logic.
    Falls back to cached data on any failure, then stooq.
    `end` defaults to today if not provided.
    """
    if end is None:
        from datetime import date
        end = date.today().strftime("%Y-%m-%d")

    # Try cache first on subsequent calls (fast path)
    cached_df = load_from_cache(ticker)
    live_failed = False

    for attempt in range(max_retries):
        try:
            print(f"[Attempt {attempt + 1}] Fetching {ticker} from yfinance ({start} to {end})...")
            df = yf.download(
                ticker, start=start, end=end,
                auto_adjust=True, session=session,
                multi_level_index=False, progress=False
            )

            if df.empty:
                raise ValueError(f"No data returned for ticker: {ticker}")

            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df.dropna(inplace=True)
            save_to_cache(ticker, df)
            return df

        except Exception as e_yf:
            print(f"[!] yfinance error (attempt {attempt + 1}/{max_retries}): {e_yf}")
            live_failed = True

            # On first failure, immediately serve cache if available (avoids long retry wait)
            if attempt == 0 and cached_df is not None:
                print(f"[!] yfinance failed — using cached data for {ticker} (will retry in background).")
                return cached_df

            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"   Waiting {wait_time}s before retry...")
                time.sleep(wait_time)

    # Try stooq as secondary fallback
    try:
        print(f"[!] yfinance exhausted retries, trying stooq...")
        df = web.DataReader(ticker, 'stooq', start=start, end=end)
        if not df.empty:
            df = df.sort_index()
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df.dropna(inplace=True)
            save_to_cache(ticker, df)
            return df
        raise ValueError(f"Stooq returned empty data for {ticker}.")
    except Exception as e_stooq:
        print(f"[!] Stooq fallback failed: {e_stooq}")

    # Last resort: cached data
    if cached_df is not None:
        print(f"[!] All live sources failed. Using cached data for {ticker} (may be slightly outdated).")
        return cached_df

    raise ValueError(
        f"All data sources failed for {ticker}. "
        "Please check your internet connection and try again."
    )


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
            print(f"[OK] Got live price: {price:.2f}")
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
    Handles both old schema (item['title']) and new schema (item['content']['title']).
    Returns empty list if unable to fetch.
    """
    try:
        stock = yf.Ticker(ticker, session=session)
        news = stock.news

        headlines = []

        if news:
            for item in news[:max_items]:
                title = ""

                # New yfinance schema: title nested under 'content'
                if "content" in item and isinstance(item["content"], dict):
                    title = item["content"].get("title", "")
                    # Filter out non-article content types (videos, etc.)
                    content_type = item["content"].get("contentType", "")
                    if content_type and content_type.lower() not in ("story", "article", ""):
                        continue

                # Old yfinance schema fallback
                if not title:
                    title = item.get("title", "")

                if title:
                    headlines.append(title)

        print(f"[Sentiment] Fetched {len(headlines)} headlines for {ticker}")
        return headlines
    except Exception as e:
        print(f"[!] Warning: Could not fetch news for {ticker}: {e}")
        return []