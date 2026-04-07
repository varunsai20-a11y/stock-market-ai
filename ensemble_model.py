"""
ensemble_model.py
-----------------
Weighted soft-voting ensemble that combines LSTM and XGBoost predictions.

Each model outputs a probability vector [P_sell, P_hold, P_buy].
We compute a weighted average and treat the argmax as the final signal.

The ensemble confidence is the max of the blended probability vector,
which is typically higher and more calibrated than either model alone.
"""

import numpy as np
import pandas as pd

from lstm_model import train_lstm_model, forecast_next_price_lstm, SEQ_LEN
from xgb_model   import train_xgb_model, predict_proba_xgb, xgb_predict_from_sequence
from models      import ALL_FEATURE_COLS


ACTION_MAP = {0: "Sell", 1: "Buy"}   # Binary: 0=Down, 1=Up
TEMPERATURE = 0.5                      # Sharpens both models' probabilities


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_ensemble(ticker: str = "AAPL",
                   start:  str = "2020-01-01",
                   end=None):
    from datetime import date
    if end is None:
        end = date.today().strftime("%Y-%m-%d")
    """
    Train both LSTM and XGBoost on the same ticker/date-range.

    Returns
    -------
    lstm_model   : trained Keras model
    xgb_model    : trained XGBoost Booster
    df           : feature-engineered DataFrame (used for inference later)
    lstm_metrics : dict
    xgb_metrics  : dict
    combined_metrics : dict — shows both side-by-side
    lstm_actual  : np.ndarray (test-set ground truth, from LSTM split)
    lstm_predicted : np.ndarray (LSTM test-set predictions)
    """
    print(f"\n{'='*50}")
    print(f"[Ensemble] Training LSTM for {ticker}…")
    print(f"{'='*50}")
    lstm_model, df, lstm_metrics, lstm_actual, lstm_predicted = train_lstm_model(
        ticker=ticker, start=start, end=end
    )

    print(f"\n{'='*50}")
    print(f"[Ensemble] Training XGBoost for {ticker}…")
    print(f"{'='*50}")
    xgb_bst, xgb_scaler, xgb_metrics = train_xgb_model(
        ticker=ticker, start=start, end=end
    )

    combined_metrics = {
        # LSTM
        "LSTM Accuracy (%)":        lstm_metrics["Accuracy (%)"],
        "LSTM Weighted F1":         lstm_metrics["Weighted F1 Score"],
        # XGBoost
        "XGBoost Accuracy (%)":     xgb_metrics["Accuracy (%)"],
        "XGBoost Weighted F1":      xgb_metrics["Weighted F1 Score"],
    }

    print(f"\n[Ensemble] Training complete.")
    print(f"  LSTM:    {lstm_metrics['Accuracy (%)']:.1f}% acc  |  F1 {lstm_metrics['Weighted F1 Score']:.1f}")
    print(f"  XGBoost: {xgb_metrics['Accuracy (%)']:.1f}% acc  |  F1 {xgb_metrics['Weighted F1 Score']:.1f}")

    return (
        lstm_model, xgb_bst,
        df,
        lstm_metrics, xgb_metrics, combined_metrics,
        lstm_actual, lstm_predicted,
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def ensemble_predict(lstm_model, xgb_bst, df: pd.DataFrame, ticker: str,
                     lstm_weight: float = 0.5, xgb_weight: float = 0.5):
    """
    Produce a final ensemble prediction from the latest data in `df`.

    Returns
    -------
    predicted_action      : str  "Sell" | "Buy"
    ensemble_confidence   : float [0..1]
    blended_probs         : np.ndarray shape (2,) — [p_down, p_up]
    predicted_price_seq   : np.ndarray shape (7,)  — 7-day price forecast (LSTM head)
    lstm_probs            : np.ndarray shape (2,)
    xgb_probs             : np.ndarray shape (2,)
    """
    # ── LSTM prediction (already temperature-scaled inside forecast_next_price_lstm)
    lstm_action, lstm_conf, lstm_probs, predicted_price_seq = forecast_next_price_lstm(
        lstm_model, df, ticker
    )

    # ── XGBoost prediction — apply temperature scaling for sharpness
    _, _, xgb_probs_raw = predict_proba_xgb(xgb_bst, df, ticker)
    scaled_xgb = xgb_probs_raw / TEMPERATURE
    xgb_probs  = np.exp(scaled_xgb) / np.exp(scaled_xgb).sum()

    # ── Blend ──
    total  = lstm_weight + xgb_weight
    w_lstm = lstm_weight / total
    w_xgb  = xgb_weight  / total

    blended_probs = w_lstm * np.array(lstm_probs) + w_xgb * np.array(xgb_probs)

    predicted_class     = int(np.argmax(blended_probs))
    ensemble_confidence = float(blended_probs[predicted_class])
    predicted_action    = ACTION_MAP[predicted_class]

    return (
        predicted_action,
        ensemble_confidence,
        blended_probs,
        predicted_price_seq,
        np.array(lstm_probs),
        np.array(xgb_probs),
    )


# ---------------------------------------------------------------------------
# Backtest helper — predict for a single row inside the backtest loop
# ---------------------------------------------------------------------------

def ensemble_predict_row(lstm_model, xgb_bst,
                         df_slice: pd.DataFrame,
                         scaler_lstm, scaler_xgb,
                         ticker: str,
                         lstm_weight: float = 0.5,
                         xgb_weight:  float = 0.5):
    """
    Binary ensemble prediction for a single point inside a backtest loop.
    """
    import joblib

    # ── LSTM path ──
    seq        = df_slice[ALL_FEATURE_COLS].values
    seq_scaled = scaler_lstm.transform(seq)
    lstm_out   = lstm_model.predict(np.array([seq_scaled]), verbose=0)
    raw_lstm   = lstm_out[0][0]                           # (2,)
    # Temperature scaling
    scaled_l   = raw_lstm / TEMPERATURE
    lstm_probs = np.exp(scaled_l) / np.exp(scaled_l).sum()

    # ── XGBoost path ──
    latest_row    = df_slice[ALL_FEATURE_COLS].iloc[-1].values
    xgb_raw_probs = xgb_predict_from_sequence(xgb_bst, scaler_xgb, latest_row)  # (2,)
    scaled_x      = xgb_raw_probs / TEMPERATURE
    xgb_probs     = np.exp(scaled_x) / np.exp(scaled_x).sum()

    # ── Blend ──
    total  = lstm_weight + xgb_weight
    w_lstm = lstm_weight / total
    w_xgb  = xgb_weight  / total

    blended_probs    = w_lstm * lstm_probs + w_xgb * xgb_probs
    predicted_class  = int(np.argmax(blended_probs))
    confidence       = float(blended_probs[predicted_class])
    predicted_action = ACTION_MAP[predicted_class]

    return predicted_action, confidence, blended_probs
