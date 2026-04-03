# Deep Learning-Based Stock Price Prediction System

An end-to-end stock price prediction application that integrates Deep Learning architectures (LSTM, RNN, CNN) with an interactive Streamlit dashboard.

## 🎯 Project Overview

This system fetches historical stock price data directly from Yahoo Finance, preprocesses the time series into sequenced data (OHLCV), and trains Deep Learning models to predict stock movement. The trained models are accessed via an intuitive interactive dashboard.

### Features
- Models supported: Multi-layer LSTM, Simple RNN, and 1D-CNN temporal models.
- Interactive multi-parameter visualizations (Actual vs. Predicted, Train vs Test).
- Deep learning 7-day future forecasting using auto-regression logic.
- Clean and modular project structure (`train.py` independent from `app.py`).

## ⚙️ Technologies Used

- **Framework**: `tensorflow` / `keras`
- **Data Engineering**: `pandas`, `numpy`, `scikit-learn` (Scaling/Metrics), `yfinance` (Data provider).
- **Frontend / Dashboard**: `streamlit`, `plotly`

## 🚀 Installation & Setup

1. Assemble the core files within your target directory.
2. Ensure you have Python installed, then install all the necessary dependencies by running:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Running the Project

### 1. Training Setup
While the Streamlit app has a built-in button to train models dynamically when missing, you can pre-train baseline models (e.g. for AAPL, TSLA) manually for a smoother dashboard experience:
```bash
python train.py --tickers AAPL TSLA --epochs 15
```

### 2. Launching the Application
Execute Streamlit to serve the frontend locally:
```bash
streamlit run app.py
```
*The Streamlit web interface will open in your default browser (usually http://localhost:8501).*

## 🧠 Model Explanations

1. **LSTM (Long Short-Term Memory)**: Specifically designed to avoid the long-term dependency problem. Highly capable of extracting sequence relations spanning far back in history, equipped with Dropout layers for regularization.
2. **Simple RNN**: Fundamental recurrent architecture that handles temporal states. Retained specifically as a performance measuring basis compared against LSTM's deeper memory logic.
3. **CNN (1D Convolution)**: Excels at capturing local temporal features across immediate day ranges from the inputted OHLCV structures. Processed vectors are flattened and output properly tailored via Dense layers.

## 📊 Modules & Responsibilities
- `app.py`: Contains standard dashboard instructions (layout setup, metric columns, plotting).
- `train.py`: Command line implementation to bypass Streamlit UI for continuous robust model generation.
- `utils.py`: Modular code decoupling, including sequence definitions, preprocessing using `MinMaxScaler`, scaling inverse mechanisms, and multi-day forecasting logic.
