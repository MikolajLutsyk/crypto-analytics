import requests
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import time
from config import DB_URL, SYMBOL, INTERVAL

# === Настройки подключения к TimescaleDB ===
engine = create_engine(DB_URL)

# === Параметры ===
DAYS = 180
LIMIT = 1000  # максимум Binance API

def create_table_if_not_exists():
    """Создание таблицы если не существует"""
    create_table_query = """
    CREATE TABLE IF NOT EXISTS ohlcv (
        id SERIAL PRIMARY KEY,
        symbol TEXT NOT NULL,
        open_time TIMESTAMP NOT NULL,
        open NUMERIC,
        high NUMERIC,
        low NUMERIC,
        close NUMERIC,
        volume NUMERIC,
        quote_asset_volume NUMERIC,
        number_of_trades INTEGER,
        taker_buy_base_volume NUMERIC,
        taker_buy_quote_volume NUMERIC,
        close_time TIMESTAMP,
        sma_7 NUMERIC,
        sma_25 NUMERIC,
        rsi_14 NUMERIC,
        macd NUMERIC,
        macd_signal NUMERIC,
        volatility_7 NUMERIC,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """
    try:
        with engine.connect() as conn:
            conn.execute(text(create_table_query))
            conn.commit()
        print("✅ Таблица ohlcv создана или уже существует")
    except Exception as e:
        print(f"❌ Ошибка создания таблицы: {e}")

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
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            if not data:
                break

            all_data.extend(data)
            start_ts = data[-1][6] + 1  # close_time + 1 мс
            print(f"📦 Loaded {len(data)} candles, next: {datetime.fromtimestamp(start_ts/1000)}")
            time.sleep(0.2)  # Rate limiting
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            break

    if not all_data:
        print("❌ No data fetched")
        return pd.DataFrame()

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

    df["symbol"] = symbol
    return df

def save_to_db(df):
    if df.empty:
        print("⚠️ No data to save.")
        return
    
    # Проверяем существующие данные
    existing_query = text("""
        SELECT MAX(open_time) as last_time FROM ohlcv WHERE symbol = :symbol
    """)
    
    with engine.connect() as conn:
        result = conn.execute(existing_query, {"symbol": SYMBOL})
        last_time = result.scalar()
    
    if last_time:
        # Фильтруем только новые данные
        df = df[df['open_time'] > last_time]
        print(f"🔄 Found {len(df)} new records to insert")
    
    if df.empty:
        print("✅ No new data to insert")
        return

    df_to_insert = df[[
        "symbol", "open_time", "open", "high", "low", "close", "volume",
        "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "close_time"
    ]]
    
    try:
        df_to_insert.to_sql("ohlcv", engine, if_exists="append", index=False)
        print(f"✅ Saved {len(df_to_insert)} rows to TimescaleDB")
    except Exception as e:
        print(f"❌ Error saving to DB: {e}")

if __name__ == "__main__":
    print("🚀 Starting data collection...")
    create_table_if_not_exists()
    
    end = datetime.utcnow()
    start = end - timedelta(days=DAYS)
    print(f"📡 Fetching {SYMBOL} {INTERVAL} data for last {DAYS} days...")
    
    df = fetch_klines(SYMBOL, INTERVAL, start, end)
    if not df.empty:
        print(f"✅ Fetched {len(df)} rows: {df['open_time'].min()} → {df['open_time'].max()}")
        save_to_db(df)
    else:
        print("❌ No data fetched")