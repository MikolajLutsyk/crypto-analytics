from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class OHLCVData(BaseModel):
    timestamps: List[datetime]
    prices: List[float]
    indicators: Dict[str, List[Optional[float]]]
    volume: List[float]

class FeatureImportance(BaseModel):
    feature: str
    importance: float

class ModelMetrics(BaseModel):
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: List[List[int]]

class Prediction(BaseModel):
    current_price: float
    predicted_direction: str
    confidence: float
    top_features: List[FeatureImportance]

class TechnicalIndicators(BaseModel):
    rsi: List[Optional[float]]
    macd: List[Optional[float]]
    macd_signal: List[Optional[float]]
    sma_7: List[Optional[float]]
    sma_25: List[Optional[float]]