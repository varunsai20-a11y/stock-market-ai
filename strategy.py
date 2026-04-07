def decide_trade(sentiment_score, predicted_action, confidence=0.0):
    """
    Decide Buy / Sell / Hold based on binary ensemble probability + sentiment.
    Threshold raised to 0.65 because binary models naturally produce higher confidence.
    """
    confidence_threshold = 0.65

    if predicted_action == "Buy" and confidence > confidence_threshold and sentiment_score > -0.1:
        return "Buy", f"Ensemble signals Buy (conf: {confidence:.0%}) & sentiment ok ({sentiment_score:.2f})."

    elif predicted_action == "Sell" and confidence > confidence_threshold and sentiment_score < 0.1:
        return "Sell", f"Ensemble signals Sell (conf: {confidence:.0%}) & sentiment ok ({sentiment_score:.2f})."

    else:
        return "Hold", f"Signal: {predicted_action} (conf: {confidence:.0%}), sent: {sentiment_score:.2f}. Waiting for higher conviction."


def execute_trade(action, cash, holdings, current_price, transaction_cost=0.001):
    """
    Simulate trade execution.
    """
    if action == "Buy" and cash > 0:
        holdings = (cash * (1 - transaction_cost)) / current_price
        cash = 0

    elif action == "Sell" and holdings > 0:
        cash = holdings * current_price * (1 - transaction_cost)
        holdings = 0

    return cash, holdings