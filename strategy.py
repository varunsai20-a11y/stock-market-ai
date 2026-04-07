def decide_trade(sentiment_score, predicted_action, confidence=0.0):
    """
    Decide Buy / Sell / Hold based on categorical prediction probability + sentiment.
    """
    # Require model to be fairly confident
    confidence_threshold = 0.45 
    
    if predicted_action == "Buy" and confidence > confidence_threshold and sentiment_score > -0.1:
        return "Buy", f"Model signals Buy (conf: {confidence:.2f}) & sentiment is ok ({sentiment_score:.2f})."

    elif predicted_action == "Sell" and confidence > confidence_threshold and sentiment_score < 0.1:
        return "Sell", f"Model signals Sell (conf: {confidence:.2f}) & sentiment is ok ({sentiment_score:.2f})."

    else:
        return "Hold", f"Signal: {predicted_action} (conf: {confidence:.2f}), sent: {sentiment_score:.2f}. Not enough conviction to trade."


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