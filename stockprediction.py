# stockprediction.py
import datetime as dt
import math
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import streamlit as st

st.set_page_config(page_title="📊 LSTM Stock Predictor", layout="wide")

# ✅ Dynamic dates
TODAY = dt.date.today()
DEFAULT_START = TODAY - dt.timedelta(days=730)
DEFAULT_END = TODAY

st.title("📈 LSTM Stock Predictor")

# Sidebar Inputs
st.sidebar.header("⚙️ Configuration")
symbol = st.sidebar.text_input("Enter Stock Symbol:", "AAPL").upper().strip()
start = st.sidebar.date_input("Enter Start Date:", DEFAULT_START)
end = st.sidebar.date_input("Enter End Date:", DEFAULT_END)
lookback = st.sidebar.number_input("Lookback window", min_value=30, max_value=120, value=60, step=5)
epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=20, value=1, step=1)

def make_sequences(data, lookback):
    x, y = [], []
    for i in range(lookback, len(data)):
        x.append(data[i - lookback:i, 0])
        y.append(data[i, 0])
    return np.array(x).reshape(-1, lookback, 1), np.array(y)

def build_model(lookback):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(lookback, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

if st.button("Train & Predict"):
    try:
        with st.spinner(f"Fetching data for {symbol}..."):
            df = yf.download(symbol, start=start, end=end, progress=False)
        if df.empty:
            st.error("No data found. Try adjusting the date range or symbol.")
            st.stop()

        st.subheader("📊 Recent Data")
        st.dataframe(df.tail(10))

        st.subheader("📉 Close Price Trend")
        st.line_chart(df["Close"])

        # Preprocessing
        data = df[["Close"]].values.astype("float32")
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(data)
        train_len = math.ceil(len(scaled) * 0.8)
        train_data, test_data = scaled[:train_len], scaled[train_len - lookback:]

        x_train, y_train = make_sequences(train_data, lookback)
        x_test, _ = make_sequences(test_data, lookback)
        y_test = data[train_len:, :]

        with st.spinner("Training model..."):
            model = build_model(lookback)
            model.fit(x_train, y_train, epochs=epochs, batch_size=1, verbose=0)

        preds = scaler.inverse_transform(model.predict(x_test, verbose=0))
        rmse = float(np.sqrt(np.mean((preds.flatten() - y_test.flatten()) ** 2)))
        st.success(f"✅ RMSE: {rmse:.4f}")

        valid = df.iloc[train_len:].copy()
        valid["Predictions"] = preds

        st.subheader("📈 Predictions vs Actual")
        st.line_chart(valid[["Close", "Predictions"]])

        st.subheader("📋 Prediction Data (last 20 rows)")
        st.dataframe(valid.tail(20))

    except Exception as e:
        st.error(f"Error: {e}")
