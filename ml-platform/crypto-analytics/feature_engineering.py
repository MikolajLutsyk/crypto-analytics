import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
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
    """Add technical indicators to DataFrame"""
    # Basic calculations
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["price_range"] = (df["high"] - df["low"]) / (df["open"] + 1e-8)
    df["body_size"] = (df["close"] - df["open"]).abs() / (df["open"] + 1e-8)
    df["is_doji"] = ((df["close"] - df["open"]).abs() / (df["high"] - df["low"] + 1e-8)) < 0.1
    df["taker_buy_ratio_base"] = df["taker_buy_base_volume"] / (df["volume"] + 1e-8)
    df["taker_buy_ratio_quote"] = df["taker_buy_quote_volume"] / (df["quote_asset_volume"] + 1e-8)
    df["volume_ratio"] = df["volume"] / (df["volume"].rolling(24).mean() + 1e-8)
    
    for period in [5, 7, 10, 20, 25, 50]:
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
        df[f"bb_pos_{period}"] = (df["close"] - df[f"bb_low_{period}"]) / ((df[f"bb_high_{period}"] - df[f"bb_low_{period}"]) + 1e-8)
    
    for period in [5, 7, 10, 20]:
        atr = AverageTrueRange(df["high"], df["low"], df["close"], window=period)
        df[f"atr_{period}"] = atr.average_true_range()
        df[f"atr_ratio_{period}"] = df[f"atr_{period}"] / (df["close"] + 1e-8)
    
    vwap = VolumeWeightedAveragePrice(df["high"], df["low"], df["close"], df["volume"], window=14)
    df["vwap"] = vwap.volume_weighted_average_price()
    df["vwap_dev"] = (df["close"] - df["vwap"]) / (df["vwap"] + 1e-8)
    
    obv = OnBalanceVolumeIndicator(df["close"], df["volume"])
    df["obv"] = obv.on_balance_volume()
    df["obv_change"] = df["obv"].pct_change()
    
    for period in [7, 14, 21]:
        df[f"volatility_{period}"] = df["return"].rolling(window=period).std() * np.sqrt(24 * 365)
    
    print(f"✓ Added {len(df.columns)} technical features")
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
    df = df.fillna(method="ffill").fillna(0)
    return df

def save_indicators_to_db(df, symbol="BTCUSDT"):
    """Save technical indicators to database for visualization"""
    engine = create_engine(DB_URL)
    
    print(f"\n{'='*60}")
    print(f"Saving indicators to database for {symbol}")
    print(f"{'='*60}")
    
    # Check which columns we have available
    required_columns = ['sma_7', 'sma_25', 'rsi_14', 'macd', 'macd_signal', 'volatility_7']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"WARNING: Missing columns in DataFrame: {missing_columns}")
        print("Available columns:", [col for col in df.columns if any(req in col for req in required_columns)])
    
    # Count records to update
    updated_count = 0
    error_count = 0
    
    # Use batch processing for efficiency
    batch_size = 200
    total_records = len(df)
    
    print(f"Processing {total_records} records in batches of {batch_size}")
    
    for i in range(0, total_records, batch_size):
        batch_end = min(i + batch_size, total_records)
        batch_df = df.iloc[i:batch_end]
        
        print(f"  Batch {i//batch_size + 1}/{(total_records + batch_size - 1)//batch_size}: "
              f"records {i} to {batch_end}")
        
        for idx, row in batch_df.iterrows():
            try:
                # Get values safely
                sma_7 = float(row.get('sma_7', 0)) if 'sma_7' in row and not pd.isna(row.get('sma_7')) else None
                sma_25 = float(row.get('sma_25', 0)) if 'sma_25' in row and not pd.isna(row.get('sma_25')) else None
                rsi_14 = float(row.get('rsi_14', 50)) if 'rsi_14' in row and not pd.isna(row.get('rsi_14')) else 50.0
                macd = float(row.get('macd', 0)) if 'macd' in row and not pd.isna(row.get('macd')) else 0.0
                macd_signal = float(row.get('macd_signal', 0)) if 'macd_signal' in row and not pd.isna(row.get('macd_signal')) else 0.0
                
                # Calculate volatility if missing
                volatility_7 = None
                if 'volatility_7' in row and not pd.isna(row.get('volatility_7')):
                    volatility_7 = float(row['volatility_7'])
                elif 'return' in row:
                    # Calculate rolling volatility
                    try:
                        # Get recent returns for volatility calculation
                        recent_idx = max(0, df.index.get_loc(idx) - 7)
                        recent_returns = df.iloc[recent_idx:df.index.get_loc(idx) + 1]['return']
                        if len(recent_returns) > 1:
                            volatility_7 = float(recent_returns.std() * np.sqrt(24 * 365))
                    except:
                        volatility_7 = None
                
                # Update database
                update_query = text("""
                    UPDATE ohlcv 
                    SET sma_7 = COALESCE(:sma_7, sma_7),
                        sma_25 = COALESCE(:sma_25, sma_25),
                        rsi_14 = COALESCE(:rsi_14, rsi_14),
                        macd = COALESCE(:macd, macd),
                        macd_signal = COALESCE(:macd_signal, macd_signal),
                        volatility_7 = COALESCE(:volatility_7, volatility_7)
                    WHERE symbol = :symbol AND open_time = :open_time
                """)
                
                with engine.connect() as conn:
                    result = conn.execute(update_query, {
                        'sma_7': sma_7,
                        'sma_25': sma_25,
                        'rsi_14': rsi_14,
                        'macd': macd,
                        'macd_signal': macd_signal,
                        'volatility_7': volatility_7,
                        'symbol': symbol,
                        'open_time': idx
                    })
                    conn.commit()
                    
                    if result.rowcount > 0:
                        updated_count += 1
                        
            except Exception as e:
                error_count += 1
                # Print first few errors for debugging
                if error_count <= 3:
                    print(f"    Error updating {idx}: {str(e)[:80]}")
                continue
    
    print(f"\nUpdate Summary:")
    print(f"  Total records processed: {total_records}")
    print(f"  Successfully updated: {updated_count}")
    print(f"  Errors: {error_count}")
    
    # Verify the update
    print(f"\nVerifying database update...")
    check_query = text("""
        SELECT 
            COUNT(*) as total,
            COUNT(CASE WHEN sma_7 IS NOT NULL THEN 1 END) as sma_7_count,
            COUNT(CASE WHEN rsi_14 IS NOT NULL THEN 1 END) as rsi_14_count,
            COUNT(CASE WHEN macd IS NOT NULL THEN 1 END) as macd_count,
            COUNT(CASE WHEN macd_signal IS NOT NULL THEN 1 END) as macd_signal_count,
            COUNT(CASE WHEN volatility_7 IS NOT NULL THEN 1 END) as volatility_7_count,
            AVG(rsi_14) as avg_rsi,
            AVG(macd) as avg_macd
        FROM ohlcv 
        WHERE symbol = :symbol
    """)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(check_query, {'symbol': symbol})
            stats = result.fetchone()
            
        print(f"\nDatabase Statistics for {symbol}:")
        print(f"  Total records: {stats[0]}")
        print(f"  With SMA_7: {stats[1]} ({stats[1]/stats[0]*100:.1f}%)")
        print(f"  With RSI_14: {stats[2]} ({stats[2]/stats[0]*100:.1f}%)")
        print(f"  With MACD: {stats[3]} ({stats[3]/stats[0]*100:.1f}%)")
        print(f"  With MACD Signal: {stats[4]} ({stats[4]/stats[0]*100:.1f}%)")
        print(f"  With Volatility_7: {stats[5]} ({stats[5]/stats[0]*100:.1f}%)")
        print(f"  Average RSI: {stats[6]:.2f}")
        print(f"  Average MACD: {stats[7]:.6f}")
        
        # Show sample of updated records
        sample_query = text("""
            SELECT open_time, close, sma_7, rsi_14, macd
            FROM ohlcv 
            WHERE symbol = :symbol 
            AND sma_7 IS NOT NULL
            ORDER BY open_time DESC
            LIMIT 3
        """)
        
        with engine.connect() as conn:
            sample_result = conn.execute(sample_query, {'symbol': symbol})
            print(f"\nSample updated records (most recent first):")
            for row in sample_result:
                print(f"  {row[0]}: Price=${row[1]:.2f}, SMA7=${row[2]:.2f}, "
                      f"RSI={row[3]:.1f}, MACD={row[4]:.6f}")
                
    except Exception as e:
        print(f"Error verifying update: {e}")
    
    print(f"\n{'='*60}")
    print("✓ Indicators saved to database successfully!")
    print(f"{'='*60}")
    return updated_count

def add_features(df):
    df = add_technical_features(df)
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_target(df)
    df = clean_data(df)
    return df

def run_etl():
    print("1. Loading data...")
    df = load_raw_data()
    print("2. Building features...")
    df = add_features(df)
    print("3. Saving to CSV...")
    df.to_csv("data/features.csv")
    print("4. Saving indicators to DB...")
    save_indicators_to_db(df)
    print("5. Done:", df.shape)

if __name__ == "__main__":
    run_etl()