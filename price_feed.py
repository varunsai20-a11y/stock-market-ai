import yfinance as yf
import pandas as pd
import requests
import numpy as np

session = requests.Session()
session.headers.update(
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
)



def fetch_stock_data(ticker="AAPL", start="2020-01-01", end="2024-12-31"):
    """
    Fetch historical stock data using yfinance. 
    Raises an error if fetching fails natively.
    """
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, session=session, multi_level_index=False)

        if df.empty:
            raise ValueError(f"No data found for ticker: {ticker}")

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"\n[!] yfinance error for {ticker}: {e}")
        raise e


def get_live_price(ticker="AAPL"):
    """
    Fetch the latest closing price.
    """
    try:
        stock = yf.Ticker(ticker, session=session)
        hist = stock.history(period="1d")

        if hist.empty:
            raise ValueError(f"Could not fetch live price for {ticker}")

        return float(hist["Close"].iloc[-1])
    except Exception as e:
        print(f"[!] Error fetching live price for {ticker}: {e}")
        raise e


def fetch_news_headlines(ticker="AAPL", max_items=10):
    """
    Fetch recent stock-related news headlines using yfinance.
    """
    stock = yf.Ticker(ticker, session=session)
    news = stock.news

    headlines = []

    if news:
        for item in news[:max_items]:
            title = item.get("title", "")
            if title:
                headlines.append(title)

    return headlines