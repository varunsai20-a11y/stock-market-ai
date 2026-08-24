import yfinance as yf
import pandas as pd
import requests
import numpy as np

import time
import os
from pathlib import Path

session = requests.Session()
session.headers.update(
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
)

import json

CACHE_DIR = Path(__file__).parent / "data"
CACHE_DIR.mkdir(exist_ok=True)


def get_cache_path(ticker):
    """Get the cache file path for a ticker."""
    return CACHE_DIR / f"{ticker}_cached.csv"


def save_meta(ticker, is_synthetic):
    """Save metadata to a JSON file."""
    meta_path = CACHE_DIR / f"{ticker}_meta.json"
    try:
        with open(meta_path, 'w') as f:
            json.dump({"is_synthetic": is_synthetic}, f)
    except Exception as e:
        print(f"[!] Failed to save metadata for {ticker}: {e}")


def load_meta(ticker):
    """Load metadata from a JSON file."""
    meta_path = CACHE_DIR / f"{ticker}_meta.json"
    if meta_path.exists():
        try:
            with open(meta_path, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def is_ticker_synthetic(ticker):
    """Check if the ticker data is currently synthetic."""
    return load_meta(ticker).get("is_synthetic", False)


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


SYNTHETIC_DATA_FLAG = {}


def generate_synthetic_data(ticker, start, end):
    """
    Generate realistic synthetic stock data using geometric Brownian motion.
    """
    print(f"[!] Generating synthetic fallback data for {ticker} from {start} to {end}...")
    base_prices = {
        "AAPL": 75.0,
        "MSFT": 160.0,
        "GOOGL": 68.0,
        "AMZN": 95.0,
        "META": 205.0,
        "NVDA": 15.0,
        "TSLA": 28.0
    }
    base_price = base_prices.get(ticker, 100.0)
    
    idx = pd.date_range(start=start, end=end, freq='B')
    n_days = len(idx)
    if n_days == 0:
        idx = pd.date_range(end=end, periods=100, freq='B')
        n_days = len(idx)
        
    mu = 0.12
    sigma = 0.25
    dt = 1 / 252.0
    
    np.random.seed(hash(ticker) % (2**32))
    daily_returns = np.random.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n_days)
    
    price_multipliers = np.exp(np.cumsum(daily_returns))
    close_prices = base_price * price_multipliers
    
    df = pd.DataFrame(index=idx)
    df['Close'] = close_prices
    
    df['Open'] = df['Close'].shift(1)
    df.loc[df.index[0], 'Open'] = base_price
    open_noise = np.random.normal(0, 0.005, n_days)
    df['Open'] = df['Open'] * (1 + open_noise)
    
    high_multiplier = 1.0 + np.abs(np.random.normal(0.01, 0.005, n_days))
    low_multiplier = 1.0 - np.abs(np.random.normal(0.01, 0.005, n_days))
    
    df['High'] = np.maximum(df['Open'], df['Close']) * high_multiplier
    df['Low'] = np.minimum(df['Open'], df['Close']) * low_multiplier
    df['Volume'] = np.random.randint(1_000_000, 50_000_000, n_days)
    
    df.index.name = 'Date'
    return df


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

            if df.empty or 'Close' not in df.columns:
                raise ValueError(f"No valid data returned for ticker: {ticker}")

            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df.dropna(inplace=True)
            save_to_cache(ticker, df)
            save_meta(ticker, False)
            SYNTHETIC_DATA_FLAG[ticker] = False
            return df

        except Exception as e_yf:
            print(f"[!] yfinance error (attempt {attempt + 1}/{max_retries}): {e_yf}")
            live_failed = True

            # On first failure, immediately serve cache if available (avoids long retry wait)
            if attempt == 0 and cached_df is not None:
                print(f"[!] yfinance failed — using cached data for {ticker} (will retry in background).")
                SYNTHETIC_DATA_FLAG[ticker] = False
                # Do not save_meta(ticker, False) here, since cached_df might have been synthetic.
                return cached_df

            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"   Waiting {wait_time}s before retry...")
                time.sleep(wait_time)



    # Last resort: cached data
    if cached_df is not None:
        print(f"[!] All live sources failed. Using cached data for {ticker} (may be slightly outdated).")
        SYNTHETIC_DATA_FLAG[ticker] = False
        return cached_df

    # Ultimate fallback: generate synthetic data
    SYNTHETIC_DATA_FLAG[ticker] = True
    synthetic_df = generate_synthetic_data(ticker, start, end)
    save_to_cache(ticker, synthetic_df)
    save_meta(ticker, True)
    return synthetic_df


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
    
    # Fallback to cache if live price fails
    cached_df = load_from_cache(ticker)
    if cached_df is not None and not cached_df.empty:
        price = float(cached_df["Close"].iloc[-1])
        print(f"[!] Live price fetch failed. Using last cached price for {ticker}: {price:.2f}")
        return price

    # Fallback to generating synthetic data to get a price
    print(f"[!] No cache available. Generating synthetic price for {ticker}...")
    from datetime import date, timedelta
    start_date = (date.today() - timedelta(days=5)).strftime("%Y-%m-%d")
    end_date = date.today().strftime("%Y-%m-%d")
    try:
        synthetic_df = generate_synthetic_data(ticker, start_date, end_date)
        price = float(synthetic_df["Close"].iloc[-1])
        return price
    except Exception as synth_e:
        print(f"[!] Failed to generate synthetic live price: {synth_e}")
    
    raise ValueError(f"Could not fetch live price for {ticker} after {max_retries} attempts and all fallbacks failed.")


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