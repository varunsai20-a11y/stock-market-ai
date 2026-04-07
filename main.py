from lstm_model import train_lstm_model, forecast_next_price_lstm
from sentiment import fetch_sentiment
from strategy import decide_trade
from price_feed import get_live_price


def main():
    ticker = "AAPL"

    print(f"\nTraining Deep Learning LSTM model for {ticker}...\n")

    model, df, metrics, actual, predicted = train_lstm_model(ticker=ticker)

    print("Model Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    predicted_action, confidence, predicted_probs = forecast_next_price_lstm(model, df)
    current_price = df["Close"].iloc[-1]

    sentiment_score, headlines = fetch_sentiment(ticker)
    action, reason = decide_trade(sentiment_score, predicted_action, confidence)

    print("\nPrediction Summary:")
    print(f"Current Price: ${current_price}")
    print(f"Predicted Signal: {predicted_action} (Confidence: {confidence:.2f})")
    print(f"Probabilities (Sell, Hold, Buy): {predicted_probs}")
    print(f"Sentiment Score: {sentiment_score}")
    print(f"Recommended Action: {action}")
    print(f"Reason: {reason}")

    print("\nRecent Headlines:")
    for h in headlines[:5]:
        print(f"- {h}")


if __name__ == "__main__":
    main()