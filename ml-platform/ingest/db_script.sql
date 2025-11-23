CREATE TABLE ohlcv (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    open_time TIMESTAMP NOT NULL,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume NUMERIC,
    quote_asset_volume NUMERIC,        -- объём в валюте котировки
    number_of_trades INTEGER,          -- количество сделок за интервал
    taker_buy_base_volume NUMERIC,     -- объём покупок (в базовой валюте)
    taker_buy_quote_volume NUMERIC,    -- объём покупок (в котируемой валюте)
    close_time TIMESTAMP,
    sma_7 NUMERIC,                     -- скользящая средняя за 7 свечей
    sma_25 NUMERIC,                    -- скользящая средняя за 25 свечей
    rsi_14 NUMERIC,                    -- RSI (Relative Strength Index)
    macd NUMERIC,                      -- MACD (Moving Average Convergence Divergence)
    macd_signal NUMERIC,               -- сигнальная линия MACD
    volatility_7 NUMERIC,              -- 7-дневная волатильность
    created_at TIMESTAMP DEFAULT NOW()
);