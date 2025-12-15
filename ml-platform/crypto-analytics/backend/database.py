from sqlalchemy import create_engine, text
import pandas as pd

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/crypto"
engine = create_engine(DATABASE_URL)

async def get_ohlcv_data(symbol: str = "BTCUSDT", days: int = 180):
    query = text("""
        SELECT * FROM ohlcv 
        WHERE symbol = :symbol 
        AND open_time >= NOW() - INTERVAL ':days days'
        ORDER BY open_time
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"symbol": symbol, "days": days})
    return df

async def get_features_data():
    """Loading data from features.csv for ML"""
    try:
        df = pd.read_csv("../data/features.csv", index_col="open_time", parse_dates=True)
        return df
    except FileNotFoundError:
        return None