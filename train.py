import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, SimpleRNN, Conv1D, MaxPooling1D, Dense, Dropout, Flatten, Input # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from utils import fetch_data, preprocess_data, create_sequences

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

def build_lstm(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=input_shape[1]) # Output sequence is matching feature count
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_rnn(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        SimpleRNN(units=50, return_sequences=True),
        Dropout(0.2),
        SimpleRNN(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=input_shape[1])
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_cnn(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(units=50, activation='relu'),
        Dense(units=input_shape[1])
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_for_ticker(ticker, start_date='2015-01-01', end_date='2024-01-01', seq_length=60, epochs=20, batch_size=32):
    print(f"\n--- Training models for {ticker} ---")
    df = fetch_data(ticker, start_date, end_date)
    if df.empty:
        print(f"No data found for {ticker}")
        return
        
    scaled_data, scaler, close_scaler, df_clean, close_idx = preprocess_data(df)
    X, y = create_sequences(scaled_data, seq_length)
    
    # Train-test split (80-20)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    model_builders = {
        'LSTM': build_lstm,
        'RNN': build_rnn,
        'CNN': build_cnn
    }
    
    for model_name, builder in model_builders.items():
        print(f"Training {model_name}...")
        model = builder((X_train.shape[1], X_train.shape[2]))
        model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=1
        )
        
        filepath = f'models/{ticker}_{model_name}.keras'
        model.save(filepath)
        print(f"Saved {model_name} to {filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stock Prediction Models")
    parser.add_argument('--tickers', nargs='+', default=['AAPL'], help='List of tickers to train on')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    args = parser.parse_args()
    
    for ticker in args.tickers:
        train_for_ticker(ticker, epochs=args.epochs)
