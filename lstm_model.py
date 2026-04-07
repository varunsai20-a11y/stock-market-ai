import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from price_feed import fetch_stock_data
from models import add_features, ALL_FEATURE_COLS
from utils import classification_metrics, ensure_directories

SEQ_LEN = 15

def create_sequences(data, target, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:(i + seq_len)])
        y.append(target[i + seq_len])
    return np.array(X), np.array(y)

def train_lstm_model(ticker="AAPL", start="2020-01-01", end="2024-12-31"):
    ensure_directories()
    
    df = fetch_stock_data(ticker, start, end)
    df = add_features(df)
    
    df_train = df.dropna(subset=["Target"])
    
    X = df_train[ALL_FEATURE_COLS].values
    y = df_train["Target"].values
    
    split_idx = int(len(X) * 0.8)

    scaler_X = MinMaxScaler()
    scaler_X.fit(X[:split_idx])
    X_scaled = scaler_X.transform(X)
    
    X_seq, y_seq = create_sequences(X_scaled, y, SEQ_LEN)
    
    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]
    
    # Building the LSTM network for Classification
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(SEQ_LEN, len(ALL_FEATURE_COLS))),
        Dropout(0.3),
        LSTM(128, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    print("Training LSTM Classifier... this may take a minute.")
    model.fit(X_train, y_train, epochs=100, batch_size=16, 
              validation_data=(X_test, y_test), 
              callbacks=[early_stop], verbose=1)
    
    predictions_prob = model.predict(X_test)
    predictions = np.argmax(predictions_prob, axis=1)
    
    metrics = classification_metrics(y_test, predictions)
    
    # Save the model and scaler
    model.save(f"models/{ticker}_lstm.h5")
    joblib.dump(scaler_X, f"models/{ticker}_scaler_X.pkl")
    
    return model, df, metrics, y_test, predictions

def forecast_next_price_lstm(model, df, ticker="AAPL"):
    scaler_X = joblib.load(f"models/{ticker}_scaler_X.pkl")
    
    latest_rows = df.iloc[-SEQ_LEN:][ALL_FEATURE_COLS].values
    latest_rows_scaled = scaler_X.transform(latest_rows)
    
    X_pred = np.array([latest_rows_scaled])
    
    predicted_probs = model.predict(X_pred)[0]
    predicted_class = int(np.argmax(predicted_probs))
    confidence = float(predicted_probs[predicted_class])
    
    # Mapping back to the string actions
    action_map = {0: "Sell", 1: "Hold", 2: "Buy"}
    predicted_action = action_map[predicted_class]
    
    return predicted_action, confidence, predicted_probs
