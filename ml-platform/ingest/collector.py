import requests
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import time

# === Настройки подключения к TimescaleDB ===
DB_URL = "postgresql+psycopg2://postgres:postgres@localhost:5432/crypto"
engine = create_engine(DB_URL)

# === Параметры ===
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
DAYS = 180
LIMIT = 1000  # максимум Binance API

def fetch_klines(symbol, interval, start_date, end_date):
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)

    while start_ts < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": LIMIT,
            "startTime": start_ts
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if not data:
            break

        all_data.extend(data)
        start_ts = data[-1][6] + 1  # close_time + 1 мс
        print(f"📦 Loaded {len(data)} candles, next: {datetime.fromtimestamp(start_ts/1000)}")
        time.sleep(0.2)

    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ])

    # Преобразование типов
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    numeric_cols = ["open", "high", "low", "close", "volume",
                    "quote_asset_volume", "taker_buy_base_volume", "taker_buy_quote_volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["number_of_trades"] = df["number_of_trades"].astype(int)

    # Добавляем индикаторы
    df["sma_7"] = df["close"].rolling(7).mean()
    df["sma_25"] = df["close"].rolling(25).mean()

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    df["volatility_7"] = df["close"].rolling(7).std()

    df["symbol"] = symbol
    return df


def save_to_db(df):
    if df.empty:
        print("⚠️ No data to save.")
        return
    df_to_insert = df[[
        "symbol", "open_time", "open", "high", "low", "close", "volume",
        "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume",
        "close_time", "sma_7", "sma_25", "rsi_14", "macd", "macd_signal", "volatility_7"
    ]]
    df_to_insert.to_sql("ohlcv", engine, if_exists="append", index=False)
    print(f"✅ Saved {len(df_to_insert)} rows to TimescaleDB")


if __name__ == "__main__":
    end = datetime.utcnow()
    start = end - timedelta(days=DAYS)
    print(f"📡 Fetching {SYMBOL} {INTERVAL} data for last {DAYS} days...")
    df = fetch_klines(SYMBOL, INTERVAL, start, end)
    print(f"✅ Fetched {len(df)} rows: {df['open_time'].min()} → {df['open_time'].max()}")
    save_to_db(df)
