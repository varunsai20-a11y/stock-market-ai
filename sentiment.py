from textblob import TextBlob
from price_feed import fetch_news_headlines


def analyze_sentiment_from_headlines(headlines):
    """
    Compute average sentiment score from headlines.
    """
    if not headlines:
        return 0.0, ["No recent headlines found. Sentiment assumed neutral."]

    finance_up = ["record", "surge", "beat", "rally", "upgrade", "outperform", "soar", "jump", "climb", "high", "gain", "rise", "profit", "bullish", "buy", "positive"]
    finance_down = ["miss", "drop", "plunge", "downgrade", "underperform", "slump", "fall", "low", "sink", "loss", "decline", "bearish", "sell", "negative", "concern"]

    scores = []
    for headline in headlines:
        polarity = TextBlob(headline).sentiment.polarity
        
        lower_h = headline.lower()
        if any(w in lower_h for w in finance_up):
            polarity += 0.3
        if any(w in lower_h for w in finance_down):
            polarity -= 0.3
            
        scores.append(max(-1.0, min(1.0, polarity)))

    avg_sentiment = sum(scores) / len(scores)
    return round(avg_sentiment, 3), headlines


def fetch_sentiment(ticker="AAPL"):
    """
    Fetch news headlines and compute sentiment.
    """
    headlines = fetch_news_headlines(ticker)
    sentiment_score, used_headlines = analyze_sentiment_from_headlines(headlines)
    return sentiment_score, used_headlines