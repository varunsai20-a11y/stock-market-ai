import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from price_feed import fetch_stock_data
from models import add_features, ALL_FEATURE_COLS
from utils import classification_metrics, ensure_directories

SEQ_LEN = 15

def create_sequences(data, target_class, target_price_7d, seq_len):
    X, y_class, y_price = [], [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:(i + seq_len)])
        y_class.append(target_class[i + seq_len - 1]) # Target for next day
        y_price.append(target_price_7d[i + seq_len - 1])
    return np.array(X), np.array(y_class), np.array(y_price)

def train_lstm_model(ticker="AAPL", start="2020-01-01", end=None):
    from datetime import date
    if end is None:
        end = date.today().strftime("%Y-%m-%d")
    ensure_directories()
    
    df = fetch_stock_data(ticker, start, end)
    df = add_features(df)
    
    PRICE_COLS = [f"Target_Price_{i}" for i in range(1, 8)]
    
    df_train = df.dropna(subset=["Target"] + PRICE_COLS)
    
    X = df_train[ALL_FEATURE_COLS].values
    y_class = df_train["Target"].values
    y_price = df_train[PRICE_COLS].values
    
    split_idx = int(len(X) * 0.8)

    scaler_X = MinMaxScaler()
    scaler_X.fit(X[:split_idx])
    X_scaled = scaler_X.transform(X)
    
    X_seq, y_class_seq, y_price_seq = create_sequences(X_scaled, y_class, y_price, SEQ_LEN)
    
    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_class_train, y_class_test = y_class_seq[:split], y_class_seq[split:]
    y_price_train, y_price_test = y_price_seq[:split], y_price_seq[split:]
    
    # Dual-Head Multi-Task LSTM Network (Binary classification + Price regression)
    input_layer = Input(shape=(SEQ_LEN, len(ALL_FEATURE_COLS)))
    
    x = LSTM(128, return_sequences=True)(input_layer)
    x = Dropout(0.2)(x)
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    
    shared_dense = Dense(64, activation='relu')(x)
    
    # Binary: 0=Down, 1=Up
    out_class = Dense(2, activation='softmax', name='class_output')(shared_dense)
    out_price = Dense(7, activation='linear', name='price_output')(shared_dense)
    
    model = Model(inputs=input_layer, outputs=[out_class, out_price])
    
    model.compile(
        optimizer='adam', 
        loss={'class_output': 'sparse_categorical_crossentropy', 'price_output': 'mse'},
        metrics={'class_output': 'accuracy'}
    )
    
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    print("Training Multi-Head LSTM... this may take a minute.")
    model.fit(
        X_train, {'class_output': y_class_train, 'price_output': y_price_train},
        validation_data=(X_test, {'class_output': y_class_test, 'price_output': y_price_test}),
        epochs=100, batch_size=16, 
        callbacks=[early_stop], verbose=1
    )
    
    predictions_out = model.predict(X_test)
    predictions_prob = predictions_out[0]
    predictions = np.argmax(predictions_prob, axis=1)
    
    metrics = classification_metrics(y_class_test, predictions)
    
    # Save the model and scaler
    model.save(f"models/{ticker}_lstm.h5")
    joblib.dump(scaler_X, f"models/{ticker}_scaler_X.pkl")
    
    return model, df, metrics, y_class_test, predictions

def forecast_next_price_lstm(model, df, ticker="AAPL", temperature=0.5):
    scaler_X = joblib.load(f"models/{ticker}_scaler_X.pkl")
    
    latest_rows = df.iloc[-SEQ_LEN:][ALL_FEATURE_COLS].values
    latest_rows_scaled = scaler_X.transform(latest_rows)
    
    X_pred = np.array([latest_rows_scaled])
    
    predicted_out = model.predict(X_pred)
    raw_logits = predicted_out[0][0]           # shape (2,)
    predicted_price_sequence = predicted_out[1][0]
    
    # Temperature scaling — T < 1 sharpens the distribution (higher confidence)
    scaled = raw_logits / temperature
    predicted_probs = np.exp(scaled) / np.exp(scaled).sum()
    
    predicted_class  = int(np.argmax(predicted_probs))
    confidence       = float(predicted_probs[predicted_class])
    
    # Binary: 0 = Down (Sell signal), 1 = Up (Buy signal)
    action_map       = {0: "Sell", 1: "Buy"}
    predicted_action = action_map[predicted_class]
    
    return predicted_action, confidence, predicted_probs, predicted_price_sequence
