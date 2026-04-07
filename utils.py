import os
import numpy as np


def ensure_directories():
    folders = ["data", "models", "outputs", "screenshots"]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)


from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def classification_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    # Using zero_division=0 to prevent warnings
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    return {
        "Accuracy (%)": round(acc * 100, 2),
        "Weighted F1 Score": round(f1 * 100, 2),
        "Confusion Matrix (Sell, Hold, Buy)": cm.tolist()
    }


def max_drawdown(portfolio_values):
    portfolio_values = np.array(portfolio_values)
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    return round(drawdown.min() * 100, 2)


def sharpe_ratio(returns, risk_free_rate=0):
    returns = np.array(returns)

    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate
    return round(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252), 2)