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
    """
    Get raw OHLCV from yfinance. We keep group_by='ticker' so yfinance may return a MultiIndex.
    """
    return yf.download(
        symbol,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,   # keep Close and Adj Close separate
        group_by="ticker",
        threads=True,
    )

def extract_close(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Return a single-column DataFrame with column 'Close' (index = dates),
    handling all common yfinance shapes:

    1) Single-level columns with 'Close'
    2) MultiIndex: level 0 = field ('Open','High',...,'Close'); level 1 = ticker
    3) MultiIndex: level 0 = ticker; level 1 = field
    4) Flattened weirdness -> look for sensible fallbacks like 'Close_AAPL', 'AAPL_Close', 'Adj Close', etc.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # Case A: already single-level with 'Close'
    if not isinstance(df.columns, pd.MultiIndex):
        cols = {c: c.title() for c in df.columns}
        df = df.rename(columns=cols)
        if "Close" in df.columns:
            return df[["Close"]]
        if "Adj Close" in df.columns:
            out = df[["Adj Close"]].rename(columns={"Adj Close": "Close"})
            return out
        return pd.DataFrame()  # no usable close

    # Case B: MultiIndex
    # Try level 0 as field names
    try:
        if "Close" in df.columns.get_level_values(0):
            sub = df.xs("Close", axis=1, level=0)  # columns are tickers (or a single series)
            if isinstance(sub, pd.Series):
                return sub.to_frame("Close")
            if symbol in sub.columns:
                return sub[[symbol]].rename(columns={symbol: "Close"})
            # If only one column, just call it Close
            if sub.shape[1] == 1:
                return sub.rename(columns={sub.columns[0]: "Close"})
    except Exception:
        pass

    # Try level -1 (last level) as field names
    try:
        if "Close" in df.columns.get_level_values(-1):
            sub = df.xs("Close", axis=1, level=-1)  # columns are tickers
            if isinstance(sub, pd.Series):
                return sub.to_frame("Close")
            if symbol in sub.columns:
                return sub[[symbol]].rename(columns={symbol: "Close"})
            if sub.shape[1] == 1:
                return sub.rename(columns={sub.columns[0]: "Close"})
    except Exception:
        pass

    # Try the inverse: slice by ticker level if present
    for lvl in (0, -1):
        try:
            if symbol in df.columns.get_level_values(lvl):
                sub = df.xs(symbol, axis=1, level=lvl)  # columns are fields
                # Normalize capitalization
                sub = sub.rename(columns={c: c.title() for c in sub.columns})
                if "Close" in sub.columns:
                    return sub[["Close"]]
                if "Adj Close" in sub.columns:
                    return sub[["Adj Close"]].rename(columns={"Adj Close": "Close"})
        except Exception:
            pass

    # Fallback: flatten and search
    flat = df.copy()
    flat.columns = ["_".join([str(x) for x in tup if x]) for tup in flat.columns.to_flat_index()]
    candidates = [
        f"Close_{symbol}",
        f"{symbol}_Close",
        "Close",
        "close",
        f"Adj Close_{symbol}",
        f"{symbol}_Adj Close",
        "Adj Close",
        "adj close",
    ]
    for cand in candidates:
        if cand in flat.columns:
            return flat[[cand]].rename(columns={cand: "Close"})

    return pd.DataFrame()

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
    if not symbol:
        st.error("Please enter a stock symbol.")
        st.stop()
    if start >= end:
        st.error("Start date must be before end date.")
        st.stop()

    raw = fetch_prices(symbol, start, end)
    if raw is None or raw.empty:
        st.error("No data returned. Try widening the date range or check the symbol.")
        st.stop()

    close_df = extract_close(raw, symbol)
    if close_df is None or close_df.empty or "Close" not in close_df.columns:
        st.error("Could not locate a usable 'Close' column from the downloaded data.")
        st.write("Raw columns snapshot:", list(raw.columns))
        st.stop()

    st.subheader("Recent Data")
    st.dataframe(close_df.tail(10))

    st.subheader("Close Price Trend")
    # Always a one-column DataFrame named 'Close' → safe for Streamlit
    st.line_chart(close_df)

    # Prepare data
    prices = close_df[["Close"]].astype("float32").values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(prices)

    # Sanity on lengths
    train_len = math.ceil(len(scaled) * 0.8)
    min_needed = max(lookback + 5, 40)
    if len(scaled) < min_needed:
        st.warning(f"Not enough data for lookback={lookback}. Add more days or reduce lookback.")
        st.stop()
    if train_len <= lookback or len(scaled) - train_len < 5:
        st.warning("Range too short for this lookback. Extend dates or reduce lookback.")
        st.stop()

    # Sequences
    X_train, y_train = make_sequences(scaled[:train_len], lookback)
    X_test, _ = make_sequences(scaled[train_len - lookback :], lookback)
    y_test = prices[train_len:, :]

    # Train
    model = build_model(lookback)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Predict
    preds = scaler.inverse_transform(model.predict(X_test, verbose=0))

    # Evaluate
    rmse = float(np.sqrt(np.mean((preds.flatten() - y_test.flatten()) ** 2)))
    st.success(f"RMSE: {rmse:,.4f}")

    # Plot predictions vs actual
    valid_idx = close_df.index[train_len:]
    valid = pd.DataFrame({"Close": y_test.flatten(), "Predictions": preds.flatten()}, index=valid_idx)

    st.subheader("Predictions vs Actual")
    st.line_chart(valid[["Close", "Predictions"]])

    st.subheader("Prediction Data (last 20)")
    st.dataframe(valid.tail(20))
