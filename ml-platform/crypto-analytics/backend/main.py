from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
from datetime import datetime, timedelta

from database import get_ohlcv_data, get_features_data
from models import OHLCVData, FeatureImportance, ModelMetrics, Prediction, TechnicalIndicators
from ml_service import ml_service

app = FastAPI(
    title="Crypto Analytics API",
    description="API for visualization of crypto data and ML model",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Crypto Analytics API"}

@app.get("/api/ohlcv/{symbol}", response_model=OHLCVData)
async def get_ohlcv(symbol: str = "BTCUSDT", days: int = 90):
    try:
        df = await get_ohlcv_data(symbol, days)
        
        # Main data
        timestamps = df['open_time'].tolist()
        prices = df['close'].tolist()
        volume = df['volume'].tolist()
        
        # Indicators
        indicators = {
            "sma_7": df['sma_7'].fillna(0).tolist(),
            "sma_25": df['sma_25'].fillna(0).tolist(),
            "rsi_14": df['rsi_14'].fillna(0).tolist(),
            "macd": df['macd'].fillna(0).tolist(),
            "macd_signal": df['macd_signal'].fillna(0).tolist(),
            "volatility_7": df['volatility_7'].fillna(0).tolist()
        }
        
        return OHLCVData(
            timestamps=timestamps,
            prices=prices,
            indicators=indicators,
            volume=volume
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/features/importance")
async def get_features_importance(top_n: int = 20):
    try:
        importance = ml_service.get_feature_importance(top_n)
        return {"features": importance}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model/metrics", response_model=ModelMetrics)
async def get_model_metrics():
    try:
        df = await get_features_data()
        if df is None:
            raise HTTPException(status_code=404, detail="Features data not found")
        
        metrics = ml_service.get_model_metrics(df)
        if metrics is None:
            raise HTTPException(status_code=404, detail="Model not trained")
        
        return ModelMetrics(**metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predict/current", response_model=Prediction)
async def get_current_prediction():
    print("=== /api/predict/current ЗАПУЩЕН ===")
    try:
        print("1. Загружаю features...")
        df = await get_features_data()
        print(f"2. Features загружены: {df is not None}")
        
        if df is None:
            print("❌ Features data not found")
            raise HTTPException(status_code=404, detail="Features data not found")
        
        print("3. Вызываю ml_service.predict_current...")
        prediction = ml_service.predict_current(df)
        
        if prediction is None:
            print("❌ Model not trained or error in prediction")
            raise HTTPException(status_code=404, detail="Model not trained")
        
        print("4. Возвращаю результат...")
        return Prediction(**prediction)
    except Exception as e:
        print(f"🔥 ОШИБКА в эндпоинте: {e}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/technical/indicators")
async def get_technical_indicators(symbol: str = "BTCUSDT", days: int = 90):
    try:
        df = await get_ohlcv_data(symbol, days)
        
        return TechnicalIndicators(
            rsi=df['rsi_14'].fillna(0).tolist(),
            macd=df['macd'].fillna(0).tolist(),
            macd_signal=df['macd_signal'].fillna(0).tolist(),
            sma_7=df['sma_7'].fillna(0).tolist(),
            sma_25=df['sma_25'].fillna(0).tolist()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)