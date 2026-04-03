import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date
from utils import fetch_data, preprocess_data, create_sequences, calculate_metrics, predict_next_days
from tensorflow.keras.models import load_model # type: ignore
import os
import subprocess

st.set_page_config(page_title="Deep Learning Stock Predictor", layout="wide", page_icon="📈")

st.title("📈 Deep Learning-Based Stock Price Prediction")
st.markdown("Predictive analytics using LSTM, Simple RNN, and CNN models.")

# --- Sidebar ---
st.sidebar.header("Configuration")
tickers = ['AAPL', 'TSLA', 'RELIANCE.NS', 'TCS.NS']
ticker = st.sidebar.selectbox("Select Stock", tickers)

model_type = st.sidebar.selectbox("Select Model", ["LSTM", "RNN", "CNN"])

START = st.sidebar.date_input("Start Date", pd.to_datetime('2015-01-01'))
END = st.sidebar.date_input("End Date", date.today())

seq_length = 60

@st.cache_data
def load_and_preprocess_data(ticker, start, end):
    df = fetch_data(ticker, start, end)
    if df.empty:
        return None, None, None, None, None
    scaled_data, scaler, close_scaler, df_clean, close_idx = preprocess_data(df)
    return scaled_data, scaler, close_scaler, df_clean, close_idx

data_load_state = st.text('Loading data...')
scaled_data, scaler, close_scaler, df_clean, close_idx = load_and_preprocess_data(ticker, START, END)

if df_clean is None:
    st.error("Error fetching data. Try a different date range or ticker.")
    st.stop()

data_load_state.text('Loading data... done!')

st.subheader('Raw Data (Recent)')
st.write(df_clean.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['Close'], name="Close Price"))
    fig.layout.update(title_text='Historical Time Series Data (Close Price)', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

plot_raw_data()

# Check if model exists
model_path = f"models/{ticker}_{model_type}.keras"
if not os.path.exists(model_path):
    st.warning(f"No pre-trained {model_type} model found for {ticker}.")
    if st.button("Train Model Now"):
        with st.spinner(f"Training {model_type} for {ticker}... This might take a few minutes."):
            # Re-train using Python subprocess
            result = subprocess.run(["python", "train.py", "--tickers", ticker, "--epochs", "10"], capture_output=True, text=True)
            if result.returncode == 0:
                st.success(f"Model {model_type} trained successfully!")
                st.rerun() # Refresh to load the newly created model
            else:
                st.error("Error during training.")
                st.write(result.stderr)
    st.stop()

# --- Model Loading & Prediction ---
@st.cache_resource
def get_model(path):
    return load_model(path)

model = get_model(model_path)

# Prepare data for prediction
X, y = create_sequences(scaled_data, seq_length)
train_size = int(len(X) * 0.8)
X_test = X[train_size:]
y_test = y[train_size:] # Actual scaled values of all features

with st.spinner("Generating test predictions..."):
    y_pred_scaled = model.predict(X_test, verbose=0)

# We need to extract the Close price.
y_test_close_scaled = y_test[:, close_idx].reshape(-1, 1)
y_pred_close_scaled = y_pred_scaled[:, close_idx].reshape(-1, 1)

y_test_close = close_scaler.inverse_transform(y_test_close_scaled)
y_pred_close = close_scaler.inverse_transform(y_pred_close_scaled)

# Metrics
rmse, mae = calculate_metrics(y_test_close, y_pred_close)
st.subheader("Model Performance")
col1, col2 = st.columns(2)
col1.metric("RMSE", f"{rmse:.2f}")
col2.metric("MAE", f"{mae:.2f}")

# Plot Test Predictions vs Actual
st.subheader("Test Data Prediction vs Actual")
test_dates = df_clean.index[seq_length + train_size:]

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=test_dates, y=y_test_close.flatten(), name="Actual Close"))
fig2.add_trace(go.Scatter(x=test_dates, y=y_pred_close.flatten(), name=f"Predicted ({model_type})"))
fig2.layout.update(title_text='Actual vs Predicted Close Price', xaxis_rangeslider_visible=True)
st.plotly_chart(fig2, use_container_width=True)

# --- Next 7 Days Prediction ---
st.subheader("Future 7-Day Forecast")
last_sequence = scaled_data[-seq_length:]

with st.spinner("Generating future forecast..."):
    future_predictions = predict_next_days(model, last_sequence, scaler, days=7)
    future_close_preds = future_predictions[:, close_idx]

future_dates = pd.date_range(df_clean.index[-1], periods=8)[1:] # next 7 days (BDays might be better but standard calendar is fine for demo)

fig3 = go.Figure()
# Show last 30 days of actual data plus the 7 days prediction
past_30_dates = df_clean.index[-30:]
past_30_close = df_clean['Close'].values[-30:]

fig3.add_trace(go.Scatter(x=past_30_dates, y=past_30_close, name="Past Close", line=dict(color='blue')))
fig3.add_trace(go.Scatter(x=future_dates, y=future_close_preds, name="Future Prediction", line=dict(color='orange', dash='dot')))
fig3.layout.update(title_text='7-Day Forecast', xaxis_rangeslider_visible=True)
st.plotly_chart(fig3, use_container_width=True)

st.write("Last Predicted Prices for the next 7 days:")
pred_df = pd.DataFrame({'Date': future_dates.date, 'Predicted Close': future_close_preds})
pred_df.set_index('Date', inplace=True)
st.dataframe(pred_df)
