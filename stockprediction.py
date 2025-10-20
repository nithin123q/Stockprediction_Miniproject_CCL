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

# ---------- Streamlit setup ----------
st.set_page_config(page_title="📈 LSTM Stock Predictor", layout="wide")
st.title("📈 LSTM Stock Predictor")

TODAY = dt.date.today()
DEFAULT_START = TODAY - dt.timedelta(days=730)  # last 2 years
DEFAULT_END = TODAY

# ---------- Helpers ----------
def fetch_prices(symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    # keep Close/Adj Close separate; group_by='ticker' gives MultiIndex for single/mtl tickers
    df = yf.download(
        symbol,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        group_by="ticker",
        threads=True,
    )
    return df

def normalize_single_ticker(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Make sure we end up with single-level columns that include 'Close'.
    Handles cases like ('Close','AAPL') or ('AAPL','Close').
    """
    if df is None or df.empty:
        return df

    # If yfinance returned a MultiIndex, slice down to this ticker
    if isinstance(df.columns, pd.MultiIndex):
        lvls = df.columns.names
        # Common case from yfinance: ('Close', 'AAPL') with names ('Price', 'Ticker')
        # If ticker appears in the LAST level:
        if symbol in df.columns.get_level_values(-1):
            df = df.xs(symbol, axis=1, level=-1)
        # Else if ticker in the FIRST level:
        elif symbol in df.columns.get_level_values(0):
            df = df.xs(symbol, axis=1, level=0)
        else:
            # Flatten as a fallback
            df.columns = ["_".join([str(x) for x in tup if x]) for tup in df.columns.to_flat_index()]

    # Normalize capitalization (e.g., 'adj close' -> 'Adj Close')
    df = df.rename(columns={c: c.title() for c in df.columns})

    # If Close missing but Adj Close present, use it
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    return df

def make_sequences(series: np.ndarray, lookback: int = 60):
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i - lookback : i, 0])
        y.append(series[i, 0])
    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y)
    return X, y

def build_model(timesteps: int):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(timesteps, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# ---------- Sidebar ----------
with st.sidebar:
    symbol = st.text_input("Enter Stock Symbol:", "AAPL").upper().strip()
    start = st.date_input("Enter Start Date:", DEFAULT_START)
    end = st.date_input("Enter End Date:", DEFAULT_END)
    lookback = st.number_input("Lookback window", min_value=20, max_value=180, value=60, step=5)
    epochs = st.number_input("Epochs", min_value=1, max_value=50, value=1, step=1)
    batch_size = st.number_input("Batch size", min_value=1, max_value=128, value=1, step=1)
    run = st.button("Train & Predict")

# ---------- Main ----------
if run:
    # Basic guards
    if not symbol:
        st.error("Please enter a stock symbol.")
        st.stop()
    if start >= end:
        st.error("Start date must be before end date.")
        st.stop()

    # Download
    df_raw = fetch_prices(symbol, start, end)
    if df_raw is None or df_raw.empty:
        st.error("No data returned. Try widening the date range or check the symbol.")
        st.stop()

    # Normalize columns → ensure 'Close'
    df = normalize_single_ticker(df_raw.copy(), symbol)
    if df is None or df.empty or "Close" not in df.columns:
        st.error("Could not find a 'Close' column after normalization.")
        st.write("Columns found:", list(df.columns))
        st.stop()

    # Show recent data & basic chart
    st.subheader("Recent Data")
    st.dataframe(df.tail(10))

    st.subheader("Close Price Trend")
    # IMPORTANT: plot a Series, not a 2-D slice
    st.line_chart(df["Close"])

    # Prepare data for LSTM
    prices = df[["Close"]].astype("float32").values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(prices)

    # Split
    train_len = math.ceil(len(scaled) * 0.8)
    # Ensure we have enough samples vs. lookback
    min_needed = max(lookback + 5, 40)
    if len(scaled) < min_needed:
        st.warning(f"Not enough data for lookback={lookback}. Add more days or reduce lookback.")
        st.stop()
    if train_len <= lookback or len(scaled) - train_len < 5:
        st.warning("Range too short for this lookback. Extend dates or reduce lookback.")
        st.stop()

    X_train, y_train = make_sequences(scaled[:train_len], lookback)
    X_test, _ = make_sequences(scaled[train_len - lookback :], lookback)
    y_test = prices[train_len:, :]

    # Train
    model = build_model(lookback)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Predict
    preds = scaler.inverse_transform(model.predict(X_test, verbose=0))

    # Error
    rmse = float(np.sqrt(np.mean((preds.flatten() - y_test.flatten()) ** 2)))
    st.success(f"RMSE: {rmse:,.4f}")

    # Plot predictions vs actual
    valid_idx = df.index[train_len:]
    valid = pd.DataFrame({"Close": y_test.flatten(), "Predictions": preds.flatten()}, index=valid_idx)

    st.subheader("Predictions vs Actual")
    st.line_chart(valid[["Close", "Predictions"]])

    st.subheader("Prediction Data (last 20)")
    st.dataframe(valid.tail(20))
