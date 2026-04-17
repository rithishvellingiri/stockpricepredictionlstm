import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

def fetch_data(ticker, start_date, end_date):
    """Fetches historical stock data using yfinance."""
    tkr = yf.Ticker(ticker)
    df = tkr.history(start=start_date, end=end_date)
    return df

def preprocess_data(df):
    """
    Handles missing values and scales all features.
    Returns scaled features, the multi-feature scaler, the price-only scaler, and the cleaned dataframe.
    """
    df = df.copy()
    
    # Determine columns based on yfinance output (sometimes MultiIndex if downloaded creatively)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    # We will use Open, High, Low, Close, Volume as features
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    # Ensure they exist
    available_features = [f for f in features if f in df.columns]
    df = df[available_features].dropna()
    
    # Scale all features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df.values)
    
    # Create a separate scaler just for Close prices to easily inverse transform just the price metrics later
    close_idx = available_features.index('Close')
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    close_scaler.fit(df[['Close']].values)
    
    return scaled_data, scaler, close_scaler, df, close_idx

def create_sequences(dataset, seq_length):
    """
    Creates sequences for time series prediction predicting all features.
    """
    X, y = [], []
    for i in range(seq_length, len(dataset)):
        X.append(dataset[i-seq_length:i, :])
        y.append(dataset[i, :]) # Target is the next timestep (all features)
    
    return np.array(X), np.array(y)

def calculate_metrics(y_true, y_pred):
    """Calculates RMSE and MAE."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae

def predict_next_days(model, last_sequence, scaler, days=7):
    """
    Predicts the next 'days' using the trained model iteratively.
    """
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days):
        curr_seq_reshaped = np.reshape(current_sequence, (1, current_sequence.shape[0], current_sequence.shape[1]))
        next_pred = model.predict(curr_seq_reshaped, verbose=0)
        predictions.append(next_pred[0])
        
        # Append prediction and remove oldest point to slide the window
        current_sequence = np.vstack((current_sequence[1:], next_pred[0]))
            
    predictions_scaled = np.array(predictions)
    # Inverse transform to get actual values
    predictions_actual = scaler.inverse_transform(predictions_scaled)
    
    return predictions_actual

def calculate_eda_metrics(df):
    """
    Calculates various EDA technical indicators and returns a new dataframe with indicators.
    """
    eda_df = df.copy()
    
    # Simple Moving Averages
    eda_df['SMA_50'] = eda_df['Close'].rolling(window=50).mean()
    eda_df['SMA_200'] = eda_df['Close'].rolling(window=200).mean()
    
    # Daily Returns
    eda_df['Daily_Return'] = eda_df['Close'].pct_change()
    
    # Cumulative Returns
    eda_df['Cumulative_Return'] = (1 + eda_df['Daily_Return'].fillna(0)).cumprod()
    
    return eda_df
