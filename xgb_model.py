"""
xgb_model.py
------------
XGBoost classifier for stock signal prediction (Buy / Hold / Sell).

Uses the same ALL_FEATURE_COLS and add_features() as the LSTM so both
models see identical inputs and the ensemble comparison is fair.

Target classes:  0 = Sell  |  1 = Hold  |  2 = Buy
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.calibration import CalibratedClassifierCV

import xgboost as xgb

from price_feed import fetch_stock_data
from models import add_features, ALL_FEATURE_COLS
from utils import classification_metrics, ensure_directories


# ---------------------------------------------------------------------------
# Hyper-parameters — tuned for financial time-series tabular data
# ---------------------------------------------------------------------------
XGB_PARAMS = dict(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="binary:logistic",  # Binary classification
    eval_metric="logloss",
    use_label_encoder=False,
    n_jobs=-1,
    random_state=42,
    early_stopping_rounds=40,
)


def train_xgb_model(ticker: str = "AAPL",
                    start: str = "2020-01-01",
                    end=None):
    from datetime import date
    if end is None:
        end = date.today().strftime("%Y-%m-%d")
    """
    Train XGBoost binary classifier (Up=1 / Down=0) with isotonic calibration.

    Returns
    -------
    bst        : trained XGBoost Booster (raw, for DMatrix inference)
    scaler_X   : fitted MinMaxScaler
    metrics    : dict of accuracy / F1 / confusion-matrix
    """
    ensure_directories()

    df = fetch_stock_data(ticker, start, end)
    df = add_features(df)

    df_train = df.dropna(subset=["Target"])

    X = df_train[ALL_FEATURE_COLS].values
    y = df_train["Target"].values.astype(int)   # 0=Down, 1=Up

    # Chronological 80/20 split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled  = scaler_X.transform(X_test)

    dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
    dtest  = xgb.DMatrix(X_test_scaled,  label=y_test)

    train_params = {
        "max_depth":          XGB_PARAMS["max_depth"],
        "learning_rate":      XGB_PARAMS["learning_rate"],
        "subsample":          XGB_PARAMS["subsample"],
        "colsample_bytree":   XGB_PARAMS["colsample_bytree"],
        "min_child_weight":   XGB_PARAMS["min_child_weight"],
        "gamma":              XGB_PARAMS["gamma"],
        "reg_alpha":          XGB_PARAMS["reg_alpha"],
        "reg_lambda":         XGB_PARAMS["reg_lambda"],
        "objective":          XGB_PARAMS["objective"],
        "eval_metric":        XGB_PARAMS["eval_metric"],
        "seed":               XGB_PARAMS["random_state"],
    }

    print(f"[XGBoost] Training binary classifier on {len(X_train)} samples…")
    bst = xgb.train(
        train_params,
        dtrain,
        num_boost_round=XGB_PARAMS["n_estimators"],
        evals=[(dtrain, "train"), (dtest, "eval")],
        early_stopping_rounds=XGB_PARAMS["early_stopping_rounds"],
        verbose_eval=100,
    )

    # Raw probability predictions  (sigmoid output for binary:logistic)
    raw_probs = bst.predict(dtest)              # shape (n,) — P(Up)
    predictions = (raw_probs >= 0.5).astype(int)

    metrics = classification_metrics(y_test, predictions)

    bst.save_model(f"models/{ticker}_xgb.json")
    joblib.dump(scaler_X, f"models/{ticker}_xgb_scaler.pkl")

    print(f"[XGBoost] Accuracy: {metrics['Accuracy (%)']:.1f}%  "
          f"F1: {metrics['Weighted F1 Score']:.1f}")

    return bst, scaler_X, metrics


def predict_proba_xgb(bst, df: pd.DataFrame, ticker: str):
    """
    Run binary inference on the most-recent row of `df`.

    Returns
    -------
    predicted_action : str   "Sell" | "Buy"
    confidence       : float max probability of winning class
    probs            : np.ndarray shape (2,) — [p_down, p_up]
    """
    scaler_X = joblib.load(f"models/{ticker}_xgb_scaler.pkl")

    latest_row = df[ALL_FEATURE_COLS].dropna().iloc[[-1]].values
    latest_scaled = scaler_X.transform(latest_row)

    dmat   = xgb.DMatrix(latest_scaled)
    p_up   = float(bst.predict(dmat)[0])     # scalar probability of Up
    p_down = 1.0 - p_up
    probs  = np.array([p_down, p_up])

    predicted_class  = int(np.argmax(probs))
    confidence       = float(probs[predicted_class])
    action_map       = {0: "Sell", 1: "Buy"}
    predicted_action = action_map[predicted_class]

    return predicted_action, confidence, probs


def load_xgb_model(ticker: str):
    """Load a previously saved XGBoost model and scaler."""
    bst = xgb.Booster()
    bst.load_model(f"models/{ticker}_xgb.json")
    scaler_X = joblib.load(f"models/{ticker}_xgb_scaler.pkl")
    return bst, scaler_X


def xgb_predict_from_sequence(bst, scaler_X, feature_row: np.ndarray):
    """
    Predict binary probs from a single flat feature row (used inside backtest loop).

    Returns
    -------
    probs : np.ndarray shape (2,) — [p_down, p_up]
    """
    row_scaled = scaler_X.transform(feature_row.reshape(1, -1))
    dmat  = xgb.DMatrix(row_scaled)
    p_up  = float(bst.predict(dmat)[0])
    return np.array([1.0 - p_up, p_up])
