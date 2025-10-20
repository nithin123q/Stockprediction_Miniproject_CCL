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

st.set_page_config(page_title="LSTM Stock Predictor", layout="wide")

# ---------- Date defaults (latest) ----------
TODAY = dt.date.today()
DEFAULT_START = TODAY - dt.timedelta(days=730)  # last 2 years
DEFAULT_END = TODAY

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def fetch_prices(symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    Fetch OHLCV data. yfinance can return MultiIndex columns when multiple tickers,
    so we normalize below.
    """
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
    return df

def normalize_ohlc(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Ensure single-level columns and guarantee a plain 'Close' column.
    """
    if df is None or df.empty:
        return df

    # If MultiIndex columns like ('Close','AAPL'), reduce to single level
    if isinstance(df.columns, pd.MultiIndex):
        try:
            if symbol in df.columns.get_level_values(-1):
                df = df.xs(symbol, axis=1, level=-1)
            else:
                df.columns = ["_".join([str(x) for x in tup if x]) for tup in df.columns.to_flat_index()]
        except Exception:
            df.columns = ["_".join([str(x) for x in tup if x]) for tup in df.columns.to_flat_index()]

    # Normalize casing (e.g., 'close' -> 'Close')
    rename_map = {c: c.title() for c in df.columns}
    df = df.rename(columns=rename_map)

    # If only Adj Close is present, create a Close alias
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    return df

def make_sequences(series: np.ndarray, lookback: int = 60):
    x, y = [], []
    for i in range(lookback, len(series)):
        x.append(series[i - lookback:i, 0])
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

# ---------- UI ----------
st.title("📈 LSTM Stock Predictor")

c1, c2, c3 = st.columns(3)
with c1:
    symbol = st.text_input("Stock Symbol", "AAPL").upper().strip()
with c2:
    start = st.date_input("Start Date", DEFAULT_START)
with c3:
    end = st.date_input("End Date", DEFAULT_END)

c4, c5, c6 = st.columns(3)
with c4:
    lookback = st.number_input("Lookback (days)", 30, 120, 60, step=5)
with c5:
    epochs = st.number_input("Epochs", 1, 20, 1)
with c6:
    batch_size = st.number_input("Batch size", 1, 64, 1)

if st.button("Train & Predict"):
    try:
        if start >= end:
            st.error("Start date must be before end date.")
            st.stop()

        with st.spinner("Downloading data…"):
            df_raw = fetch_prices(symbol, start, end)

        if df_raw is None or df_raw.empty:
            st.error("No data returned. Check symbol and date range.")
            st.stop()

        # Normalize columns to guarantee a plain 'Close'
        df = normalize_ohlc(df_raw.copy(), symbol)
        if "Close" not in df.columns:
            st.error("Downloaded data does not contain a 'Close' column after normalization.")
            st.write("Columns received:", list(df.columns))
            st.stop()

        st.subheader("Recent Data")
        st.dataframe(df.tail(10))

        st.subheader("Close Price")
        st.line_chart(df["Close"])  # plot Series (safe)

        # Prepare data
        data = df[["Close"]].copy()
        values = data.values.astype("float32")

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)

        # 80/20 split
        train_len = math.ceil(len(scaled) * 0.8)
        if train_len <= lookback or len(scaled) - train_len < max(5, lookback):
            st.warning("Date range is too short for the chosen lookback. Extend the range or reduce lookback.")
            st.stop()

        train_data = scaled[:train_len]
        test_data = scaled[train_len - lookback:]  # overlap

        x_train, y_train = make_sequences(train_data, lookback)
        x_test, _ = make_sequences(test_data, lookback)
        y_test = values[train_len:, :]  # unscaled for metrics/plot

        with st.spinner("Training model…"):
            model = build_model(lookback)
            model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        preds = scaler.inverse_transform(model.predict(x_test, verbose=0))

        # Correct RMSE
        rmse = float(np.sqrt(np.mean((preds.flatten() - y_test.flatten()) ** 2)))
        st.success(f"RMSE: {rmse:,.4f}")

        train = data.iloc[:train_len].copy()
        valid = data.iloc[train_len:].copy()
        valid["Predictions"] = preds

        st.subheader("Train vs Valid")
        st.line_chart(pd.DataFrame({"Train": train["Close"], "Valid": valid["Close"]}))

        st.subheader("Valid vs Predictions")
        st.line_chart(valid[["Close", "Predictions"]])

        st.subheader("Validation (tail)")
        st.dataframe(valid.tail(20))

        st.caption("Tip: Increase epochs for a tighter fit. Longer lookback may need a longer date range.")

    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.stop()
