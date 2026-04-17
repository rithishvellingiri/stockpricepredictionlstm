import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date
from utils import fetch_data, preprocess_data, create_sequences, calculate_metrics, predict_next_days, calculate_eda_metrics
from tensorflow.keras.models import load_model # type: ignore
import os
import subprocess
import sys

# Custom CSS for premium glassmorphism feel
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(5px);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 5px 5px 0px 0px;
        color: #ddd;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.15) !important;
        border-bottom: 2px solid #00d2ff !important;
    }
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Deep Learning Stock Predictor", layout="wide", page_icon="📈")

st.title("📈 Deep Learning-Based Stock Price Prediction")
st.markdown("Predictive analytics using LSTM, Simple RNN, and CNN models.")

# --- Sidebar ---
st.sidebar.header("Configuration")
tickers = ['AAPL', 'TSLA', 'RELIANCE.NS', 'TCS.NS']
ticker = st.sidebar.selectbox("Select Stock", tickers)

model_type = st.sidebar.selectbox("Select Model", ["LSTM", "RNN", "CNN"])

START = st.sidebar.date_input("Start Date", date(2015, 1, 1))
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

if len(df_clean) < seq_length + 20: # Ensure enough data for sequences and test set
    st.error(f"Insufficient data for selected range. Minimum {seq_length + 20} data points required.")
    st.stop()

data_load_state.text('Loading data... done!')

st.subheader('Raw Data (Recent)')
st.write(df_clean.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['Close'], name="Close Price", line=dict(color='#00d2ff', width=2)))
    fig.update_layout(
        template="plotly_dark",
        title='Historical Time Series Data (Close Price)',
        xaxis_rangeslider_visible=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#ffffff")
    )
    st.plotly_chart(fig, use_container_width=True)

plot_raw_data()

# --- EDA Section ---
st.divider()
st.header("📊 Exploratory Data Analysis")
eda_df = calculate_eda_metrics(df_clean)

tab1, tab2, tab3, tab4 = st.tabs(["Price & MAs", "Volume Analysis", "Daily Returns", "Cumulative Growth"])

with tab1:
    st.subheader("Price with Moving Averages (50 & 200)")
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=eda_df.index, y=eda_df['Close'], name='Close Price', line=dict(color='#0077b6')))
    fig_ma.add_trace(go.Scatter(x=eda_df.index, y=eda_df['SMA_50'], name='50-day SMA', line=dict(color='#f39c12')))
    fig_ma.add_trace(go.Scatter(x=eda_df.index, y=eda_df['SMA_200'], name='200-day SMA', line=dict(color='#e74c3c')))
    fig_ma.update_layout(
        template="plotly_dark",
        title="Closing Price Trends",
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_ma, use_container_width=True)

with tab2:
    st.subheader("Trading Volume Analysis")
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(x=eda_df.index, y=eda_df['Volume'], name='Volume', marker_color='#9b59b6'))
    fig_vol.update_layout(
        template="plotly_dark",
        title="Daily Trading Volume",
        xaxis_title="Date",
        yaxis_title="Volume"
    )
    st.plotly_chart(fig_vol, use_container_width=True)

with tab3:
    st.subheader("Daily Returns Distribution")
    fig_dist = go.Figure()
    returns_data = eda_df['Daily_Return'].dropna()
    fig_dist.add_trace(go.Histogram(x=returns_data, nbinsx=100, name='Daily Returns', marker_color='#1abc9c', opacity=0.75))
    fig_dist.update_layout(
        template="plotly_dark",
        title="Volatility Distribution",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        bargap=0.05
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with tab4:
    st.subheader("Cumulative Returns (Growth of 1 unit)")
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(x=eda_df.index, y=eda_df['Cumulative_Return'], name='Cumulative Return', line=dict(color='#2ecc71', width=2), fill='tozeroy'))
    fig_cum.update_layout(
        template="plotly_dark",
        title="Growth Over Time",
        xaxis_title="Date",
        yaxis_title="Cumulative Return"
    )
    st.plotly_chart(fig_cum, use_container_width=True)

st.divider()

# Check if model exists
model_path = f"models/{ticker}_{model_type}.keras"
if not os.path.exists(model_path):
    st.warning(f"No pre-trained {model_type} model found for {ticker}.")
    if st.button("Train Model Now"):
        with st.spinner(f"Training {model_type} for {ticker}... This might take a few minutes."):
            # Re-train using Python subprocess with sys.executable for stability
            result = subprocess.run([sys.executable, "train.py", "--tickers", ticker, "--epochs", "10"], capture_output=True, text=True)
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
fig2.add_trace(go.Scatter(x=test_dates, y=y_test_close.flatten(), name="Actual Close", line=dict(color='#00d2ff')))
fig2.add_trace(go.Scatter(x=test_dates, y=y_pred_close.flatten(), name=f"Predicted ({model_type})", line=dict(color='#ff9f43', dash='dot')))
fig2.update_layout(
    template="plotly_dark",
    title_text='Actual vs Predicted Close Price',
    xaxis_rangeslider_visible=True,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
st.plotly_chart(fig2, use_container_width=True)

# --- Next 7 Days Prediction ---
st.subheader("Future 7-Day Forecast")
last_sequence = scaled_data[-seq_length:]

with st.spinner("Generating future forecast..."):
    future_predictions = predict_next_days(model, last_sequence, scaler, days=7)
    future_close_preds = future_predictions[:, close_idx]

future_dates = pd.date_range(df_clean.index[-1], periods=8)[1:] # next 7 days

fig3 = go.Figure()
# Show last 45 days of actual data plus the 7 days prediction
past_45_dates = df_clean.index[-45:]
past_45_close = df_clean['Close'].values[-45:]

fig3.add_trace(go.Scatter(x=past_45_dates, y=past_45_close, name="Past Close", line=dict(color='#00d2ff', width=2)))
fig3.add_trace(go.Scatter(x=future_dates, y=future_close_preds, name="Future Prediction", line=dict(color='#f39c12', width=3, dash='dash')))
fig3.update_layout(
    template="plotly_dark",
    title_text='7-Day Forecast',
    xaxis_rangeslider_visible=True,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
st.plotly_chart(fig3, use_container_width=True)

st.write("Last Predicted Prices for the next 7 days:")
pred_df = pd.DataFrame({'Date': future_dates.date, 'Predicted Close': future_close_preds})
pred_df.set_index('Date', inplace=True)
st.dataframe(pred_df)
