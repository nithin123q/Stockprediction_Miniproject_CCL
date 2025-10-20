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

TODAY = dt.date.today()
DEFAULT_START = TODAY - dt.timedelta(days=730)
DEFAULT_END = TODAY

def fetch_prices(symbol, start, end):
    return yf.download(
        symbol,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,   # keep Close & Adj Close explicit
        group_by="column",
        threads=True,
    )

def ensure_close_column(df, symbol):
    if isinstance(df.columns, pd.MultiIndex):
        if symbol in df.columns.get_level_values(-1):
            df = df.xs(symbol, axis=1, level=-1)
        else:
            df.columns = ["_".join([str(x) for x in tup if x]) for tup in df.columns.to_flat_index()]
    df = df.rename(columns={c: c.title() for c in df.columns})
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    return df

def make_sequences(series, lookback=60):
    x, y = [], []
    for i in range(lookback, len(series)):
        x.append(series[i-lookback:i, 0])
        y.append(series[i, 0])
    x = np.array(x).reshape(-1, lookback, 1)
    y = np.array(y)
    return x, y

def build_model(timesteps):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(timesteps, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

st.title("📈 LSTM Stock
