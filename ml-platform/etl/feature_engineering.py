import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator
from config import DB_URL, SYMBOL


def load_raw_data(symbol=SYMBOL, lookback_days=25):
    engine = create_engine(DB_URL)
    query = f"""
        SELECT open_time, open, high, low, close, volume,
               quote_asset_volume, number_of_trades,
               taker_buy_base_volume, taker_buy_quote_volume
        FROM ohlcv
        WHERE symbol = '{symbol}'
        ORDER BY open_time ASC
    """
    df = pd.read_sql(query, engine, parse_dates=["open_time"])
    df = df.set_index("open_time").asfreq("1H").fillna(method="ffill")
    if lookback_days:
        df = df.last(f"{lookback_days}D")
    return df


def add_technical_features(df):
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["price_range"] = (df["high"] - df["low"]) / (df["open"] + 1e-8)
    df["body_size"] = (df["close"] - df["open"]).abs() / (df["open"] + 1e-8)
    df["is_doji"] = ((df["close"] - df["open"]).abs() / (df["high"] - df["low"] + 1e-8)) < 0.1
    df["taker_buy_ratio_base"] = df["taker_buy_base_volume"] / (df["volume"] + 1e-8)
    df["taker_buy_ratio_quote"] = df["taker_buy_quote_volume"] / (df["quote_asset_volume"] + 1e-8)
    df["volume_ratio"] = df["volume"] / (df["volume"].rolling(24).mean() + 1e-8)
    for period in [5, 10, 20, 50]:
        df[f"sma_{period}"] = SMAIndicator(df["close"], period).sma_indicator()
        df[f"ema_{period}"] = EMAIndicator(df["close"], period).ema_indicator()
        df[f"sma_ratio_{period}"] = df["close"] / (df[f"sma_{period}"] + 1e-8) - 1
    macd_obj = MACD(df["close"], window_fast=12, window_slow=26, window_sign=9)
    df["macd"] = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()
    df["macd_diff"] = macd_obj.macd_diff()
    for period in [7, 14, 21]:
        df[f"rsi_{period}"] = RSIIndicator(df["close"], period).rsi()
    for period in [10, 20]:
        bb = BollingerBands(df["close"], window=period, window_dev=2)
        df[f"bb_high_{period}"] = bb.bollinger_hband()
        df[f"bb_low_{period}"] = bb.bollinger_lband()
        df[f"bb_middle_{period}"] = bb.bollinger_mavg()
        df[f"bb_width_{period}"] = (df[f"bb_high_{period}"] - df[f"bb_low_{period}"]) / (df[f"bb_middle_{period}"] + 1e-8)
        df[f"bb_pos_{period}"] = (df["close"] - df[f"bb_low_{period}"]) / (
            (df[f"bb_high_{period}"] - df[f"bb_low_{period}"]) + 1e-8)
    for period in [5, 10, 20]:
        atr = AverageTrueRange(df["high"], df["low"], df["close"], window=period)
        df[f"atr_{period}"] = atr.average_true_range()
        df[f"atr_ratio_{period}"] = df[f"atr_{period}"] / (df["close"] + 1e-8)
    vwap = VolumeWeightedAveragePrice(df["high"], df["low"], df["close"], df["volume"], window=14)
    df["vwap"] = vwap.volume_weighted_average_price()
    df["vwap_dev"] = (df["close"] - df["vwap"]) / (df["vwap"] + 1e-8)
    obv = OnBalanceVolumeIndicator(df["close"], df["volume"])
    df["obv"] = obv.on_balance_volume()
    df["obv_change"] = df["obv"].pct_change()
    return df


def add_time_features(df):
    df.index = pd.to_datetime(df.index)
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["day_of_month"] = df.index.day
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def add_lag_features(df, lags=[1, 2, 3, 6, 12, 24, 48]):
    for lag in lags:
        df[f"return_lag_{lag}"] = df["return"].shift(lag)
        df[f"volume_lag_{lag}"] = df["volume"].shift(lag)
        df[f"price_range_lag_{lag}"] = df["price_range"].shift(lag)
        df[f"close_lag_{lag}"] = df["close"].shift(lag)
    for window in [3, 6, 12, 24]:
        df[f"return_ma_{window}"] = df["return"].rolling(window).mean()
        df[f"return_std_{window}"] = df["return"].rolling(window).std()
        df[f"volume_ma_{window}"] = df["volume"].rolling(window).mean()
        df[f"price_range_ma_{window}"] = df["price_range"].rolling(window).mean()
    return df


def add_target(df, horizon=1):
    df["close_future"] = df["close"].shift(-horizon)
    df["future_return"] = (df["close_future"] - df["close"]) / (df["close"] + 1e-8)
    df["target_direction"] = (df["future_return"] > 0).astype(int)
    thr = df["future_return"].abs().quantile(0.6)
    df["target_3class"] = 0
    df.loc[df["future_return"] > thr, "target_3class"] = 1
    df.loc[df["future_return"] < -thr, "target_3class"] = -1
    return df


def clean_data(df):
    df = df[(df["volume"] > 0) & (df["close"] > 0)]
    df = df.fillna(method="ffill").dropna()
    return df


def add_features(df):
    df = add_technical_features(df)
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_target(df)
    df = clean_data(df)
    return df


def run_etl():
    print("📥 Loading data...")
    df = load_raw_data()
    print("⚙️ Building features...")
    df = add_features(df)
    print("💾 Saving to CSV...")
    df.to_csv("../data/features.csv")
    print("✅ Done:", df.shape)


if __name__ == "__main__":
    run_etl()
