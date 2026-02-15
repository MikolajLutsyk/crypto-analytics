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

CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_time ON ohlcv(symbol, open_time);
CREATE INDEX IF NOT EXISTS idx_ohlcv_time ON ohlcv(open_time);
