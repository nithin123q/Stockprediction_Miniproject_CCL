import datetime as dt
import math
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import streamlit as st

st.set_page_config(page_title="LSTM Stock Predictor", layout="wide")

# --- Helpers ---
def load_prices(symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, progress=False)
    return df

def make_sequences(series: np.ndarray, lookback: int = 60):
    x, y = [], []
    for i in range(lookback, len(series)):
        x.append(series[i-lookback:i, 0])
        y.append(series[i, 0])
    x = np.array(x).reshape(-1, lookback, 1)
    y = np.array(y)
    return x, y

def build_model(timesteps: int):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(timesteps, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# --- Sidebar ---
st.title("📈 LSTM Stock Predictor")
symbol = st.sidebar.text_input("Enter Stock Symbol:", value="AAPL")
start = st.sidebar.date_input("Enter Start Date:", dt.date(2020, 3, 4))
end = st.sidebar.date_input("Enter End Date:", dt.date(2021, 5, 6))
lookback = st.sidebar.number_input("Lookback window", 30, 120, 60, step=5)
epochs = st.sidebar.number_input("Epochs", 1, 20, 1)

# --- UI Actions ---
if st.button("Train & Predict"):
    with st.spinner("Downloading data and training model…"):
        df = load_prices(symbol, start, end)
        if df.empty:
            st.error("No data returned. Check symbol/dates.")
            st.stop()

        st.subheader("Recent Data")
        st.dataframe(df.tail(10))
        st.line_chart(df[["Close"]])

        data = df[["Close"]].copy()
        values = data.values

        # Scale
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)

        # Split (80/20)
        train_len = math.ceil(len(scaled) * 0.8)
        train_data = scaled[:train_len]
        test_data = scaled[train_len - lookback:]  # overlap

        # Sequences
        x_train, y_train = make_sequences(train_data, lookback)
        x_test, _ = make_sequences(test_data, lookback)
        y_test = values[train_len:, :]  # unscaled

        # Model
        model = build_model(lookback)
        model.fit(x_train, y_train, epochs=epochs, batch_size=1, verbose=0)

        # Predict & invert
        preds = scaler.inverse_transform(model.predict(x_test))

        # RMSE (fixed)
        rmse = np.sqrt(np.mean((preds.flatten() - y_test.flatten())**2))
        st.info(f"RMSE: {rmse:,.4f}")

        # Plot
        train = data.iloc[:train_len].copy()
        valid = data.iloc[train_len:].copy()
        valid["Predictions"] = preds

        st.subheader("Train vs Valid")
        st.line_chart(pd.DataFrame({"Train": train["Close"], "Valid": valid["Close"]}))
        st.subheader("Valid vs Predictions")
        st.line_chart(valid[["Close", "Predictions"]])

        st.subheader("Validation (tail)")
        st.dataframe(valid.tail(20))

        st.success("Done ✅")
