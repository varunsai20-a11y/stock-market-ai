from models import train_models, forecast_next_price
from sentiment import fetch_sentiment
from strategy import decide_trade
from price_feed import get_live_price


def main():
    # Popular tech stocks: AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA
    ticker = "AAPL"
    
    print(f"\nTraining model for {ticker}...\n")

    model, df, metrics, actual, predicted = train_model(ticker=ticker)

    print("Model Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    predicted_price, current_price, change = forecast_next_price(model, df)

    sentiment_score, headlines = fetch_sentiment(ticker)
    action, reason = decide_trade(sentiment_score, predicted_price, current_price)

    print("\nPrediction Summary:")
    print(f"Current Price: ${current_price}")
    print(f"Predicted Next Price: ${predicted_price}")
    print(f"Predicted Change: {change}%")
    print(f"Sentiment Score: {sentiment_score}")
    print(f"Recommended Action: {action}")
    print(f"Reason: {reason}")

    print("\nRecent Headlines:")
    for h in headlines[:5]:
        print(f"- {h}")


if __name__ == "__main__":
    main()