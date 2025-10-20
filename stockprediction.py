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

# ---- Dynamic default dates (latest) ----
TODAY = dt.date.today()
DEFAULT_START = TODAY - dt.timedelta(days=730)
DEFAULT_END = TODAY

# ---- Session state so widgets don't stick to old values ----
if "start_date" not in st.session_state:
    st.session_state.start_date = DEFAULT_START
if "end_date" not in st.session_state:
    st.session_state.end_date = DEFAULT_END

st.title("📈 LSTM Stock Predictor")

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def fetch_prices(symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    yfinance can return MultiIndex columns when multiple tickers or metadata are present.
    We also set auto_adjust=False so 'Close' and 'Adj Close' are explicit.
    """
    return yf.download(
        symbol,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,  # avoid future default surprises
        group_by="column",
        threads=True,
    )

def normalize_ohlc(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Ensure single-level columns and guarantee a plain 'Close' column.
    Handles: ('Close','AAPL') style MultiIndex, different casings, and Adj Close only.
    """
    if df is None or df.empty:
        return df

    # 1) If MultiIndex, try to select the level for the ticker first; otherwise flatten.
    if isinstance(df.columns, pd.MultiIndex):
        try:
            if symbol in df.columns.get_level_values(-1):
                df = df.xs(symbol, axis=1, level=-1)  # pick 'AAPL' slice
            else:
                df.columns = ["_".join([str(x) for x in tup if x]) for tup in df.columns.to_flat_index()]
        except Exception:
            df.columns = ["_".join([str(x) for x in tup if x]) for tup in df.columns.to_flat_index()]

    # 2) Standardize capitalization
    df = df.rename(columns={c: c.title() for c in df.columns})

    # 3) If missing Close, but Adj Close exists, alias it
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

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Configuration")
symbol = st.sidebar.text_input("Enter Stock Symbol:", "AAPL").upper().strip()

col_reset1, col_reset2 = st.sidebar.columns(2)
if col_reset1.button("Last 2 Years"):
    st.session_state.start_date = DEFAULT_START
    st.session_state.end_date = DEFAULT_END
if col_reset2.button("YTD"):
    st.session_state.start_date = dt.date(TODAY.year, 1, 1)
    st.session_state.end_date = DEFAULT_END

start = st.sidebar.date_input("Enter Start Date:", value=st.session_state.start_date, key="start_date")
end = st.sidebar.date_input("Enter End Date:", value=st.session_state.end_date, key="end_date")

lookback = st.sidebar.number_input("Lookback window", min_value=30, max_value=120, value=60, step=5)
epochs   = st.sidebar.number_input("Epochs", min_value=1, max_value=20, value=1, step=1)
batch_sz = st.sidebar.number_input("Batch size", min_value=1, max_value=64, value=1, step=1)

run = st.button("Train & Predict")

# ---------- Main ----------
if run:
    try:
        if start >= end:
            st.error("Start date must be before end date.")
            st.stop()

        with st.spinner(f"Fetching data for {symbol}…"):
            df_raw = fetch_prices(symbol, start, end)

        if df_raw is None or df_raw.empty:
            st.error("No data returned. Try a broader date range or a different symbol.")
            st.stop()

        df = normalize_ohlc(df_raw.copy(), symbol)

        # Debug info if needed:
        # st.write("Columns after normalize:", list(df.columns))

        if "Close" not in df.columns:
            st.error("Downloaded data lacks a 'Close' column after normalization.")
            st.write("Columns:", list(df.columns))
            st.stop()

        st.subheader("📊 Recent Data")
        st.dataframe(df.tail(10))

        st.subheader("📉 Close Price Trend")
        # IMPORTANT: plot a Series to avoid MultiIndex pitfalls
        st.line_chart(df["Close"])

        # Prepare data
        data = df[["Close"]].astype("float32").values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(data)

        # 80/20 split with sanity check for lookback
        train_len = math.ceil(len(scaled) * 0.8)
        if train_len <= lookback or len(scaled) - train_len < max(5, lookback):
            st.warning("Date range too short for this lookback. Extend range or reduce lookback.")
            st.stop()

        train_data = scaled[:train_len]
        test_data  = scaled[train_len - lookback:]

        x_train, y_train = make_sequences(train_data, lookback)
        x_test, _        = make_sequences(test_data, lookback)
        y_test = data[train_len:, :]  # unscaled

        with st.spinner("Training model…"):
            model = build_model(lookback)
            model.fit(x_train, y_train, epochs=epochs, batch_size=batch_sz, verbose=0)

        preds = scaler.inverse_transform(model.predict(x_test, verbose=0))
        rmse = float(np.sqrt(np.mean((preds.flatten() - y_test.flatten()) ** 2)))
        st.success(f"✅ RMSE: {rmse:,.4f}")

        # Build a frame aligned to original index for the validation region
        valid_index = df.index[train_len:]
        valid = pd.DataFrame({"Close": y_test.flatten(), "Predictions": preds.flatten()}, index=valid_index)

        st.subheader("📈 Predictions vs Actual")
        st.line_chart(valid[["Close", "Predictions"]])

        st.subheader("📋 Prediction Data (last 20 rows)")
        st.dataframe(valid.tail(20))

        st.caption("Tip: Increase epochs for a tighter fit. Longer lookback needs a longer date range.")

    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.stop()
