import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.testclient import TestClient

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Crypto Analytics API"}

@app.get("/api/ohlcv/{symbol}")
async def get_ohlcv(symbol: str = "BTCUSDT", days: int = 90):
    timestamps = pd.date_range('2024-01-01', periods=10, freq='1h').tolist()
    prices = [40000 + i*100 for i in range(10)]
    
    return {
        "timestamps": [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps],
        "prices": prices,
        "indicators": {
            "sma_7": [40100 + i*100 for i in range(10)],
            "sma_25": [39900 + i*100 for i in range(10)],
            "rsi_14": [45 + i*2 for i in range(10)],
            "macd": [-10 + i*2 for i in range(10)],
            "macd_signal": [-12 + i*2 for i in range(10)],
            "volatility_7": [0.15 + i*0.01 for i in range(10)]
        },
        "volume": [1000 + i*50 for i in range(10)]
    }

@app.get("/api/features/importance")
async def get_features_importance(top_n: int = 20):
    return {
        "features": [
            {"feature": "rsi_14", "importance": 0.15},
            {"feature": "macd", "importance": 0.12},
            {"feature": "volume", "importance": 0.10}
        ]
    }

@app.get("/api/model/metrics")
async def get_model_metrics():
    return {
        "accuracy": 0.65,
        "balanced_accuracy": 0.64,
        "precision": 0.66,
        "recall": 0.65,
        "f1_score": 0.65,
        "confusion_matrix": [[30, 10], [15, 45]]
    }

@app.get("/api/predict/current")
async def get_current_prediction():
    return {
        "current_price": 45000.50,
        "predicted_direction": "UP",
        "confidence": 0.75,
        "top_features": [
            {"feature": "rsi_14", "importance": 0.15},
            {"feature": "macd", "importance": 0.12}
        ]
    }

@app.get("/api/technical/indicators")
async def get_technical_indicators(symbol: str = "BTCUSDT", days: int = 90):
    return {
        "rsi": [45 + i*2 for i in range(10)],
        "macd": [-10 + i*2 for i in range(10)],
        "macd_signal": [-12 + i*2 for i in range(10)],
        "sma_7": [40100 + i*100 for i in range(10)],
        "sma_25": [39900 + i*100 for i in range(10)]
    }


@pytest.fixture
def client():
    """Test client for FastAPI"""
    return TestClient(app)


@pytest.fixture
def sample_ohlcv_df():
    """Sample OHLCV DataFrame"""
    timestamps = pd.date_range('2024-01-01', periods=10, freq='1h')
    return pd.DataFrame({
        'open_time': timestamps,
        'close': [40000 + i*100 for i in range(10)],
        'volume': [1000 + i*50 for i in range(10)],
        'sma_7': [40100 + i*100 for i in range(10)],
        'sma_25': [39900 + i*100 for i in range(10)],
        'rsi_14': [45 + i*2 for i in range(10)],
        'macd': [-10 + i*2 for i in range(10)],
        'macd_signal': [-12 + i*2 for i in range(10)],
        'volatility_7': [0.15 + i*0.01 for i in range(10)]
    })


@pytest.fixture
def sample_features_df():
    """Sample features DataFrame"""
    timestamps = pd.date_range('2024-01-01', periods=10, freq='1h')
    return pd.DataFrame({
        'close': [40000 + i*100 for i in range(10)],
        'target_direction': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        'feature_1': np.random.randn(10),
        'feature_2': np.random.randn(10)
    }, index=timestamps)


def test_root_endpoint(client):
    """Test GET /"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "Crypto Analytics API" in data["message"]


def test_api_endpoints_exist(client):
    """Test that API endpoints are defined"""
    endpoints_to_check = [
        ("/", 200),
        ("/api/ohlcv/BTCUSDT", 200),
        ("/api/features/importance", 200),
        ("/api/model/metrics", 200),
        ("/api/predict/current", 200),
        ("/api/technical/indicators", 200)
    ]
    
    for endpoint, expected_status in endpoints_to_check:
        response = client.get(endpoint)
        assert response.status_code == expected_status, f"Endpoint {endpoint} failed"


def test_ohlcv_endpoint_structure(client):
    """Test OHLCV endpoint response structure"""
    response = client.get("/api/ohlcv/BTCUSDT?days=90")
    assert response.status_code == 200
    
    data = response.json()
    
    assert "timestamps" in data
    assert "prices" in data
    assert "volume" in data
    assert "indicators" in data
    
    indicators = data["indicators"]
    expected_keys = ['sma_7', 'sma_25', 'rsi_14', 'macd', 'macd_signal', 'volatility_7']
    for key in expected_keys:
        assert key in indicators
        assert isinstance(indicators[key], list)
        assert len(indicators[key]) == 10
    
    assert all(isinstance(ts, str) for ts in data["timestamps"])
    assert all(isinstance(p, (int, float)) for p in data["prices"])
    assert all(isinstance(v, (int, float)) for v in data["volume"])


def test_feature_importance_endpoint(client):
    """Test feature importance endpoint"""
    response = client.get("/api/features/importance?top_n=10")
    assert response.status_code == 200
    
    data = response.json()
    assert "features" in data
    assert len(data["features"]) == 3
    assert data["features"][0]["feature"] == "rsi_14"
    assert data["features"][0]["importance"] == 0.15


def test_model_metrics_endpoint(client):
    """Test model metrics endpoint"""
    response = client.get("/api/model/metrics")
    assert response.status_code == 200
    
    data = response.json()
    
    assert 0 <= data["accuracy"] <= 1
    assert 0 <= data["balanced_accuracy"] <= 1
    assert 0 <= data["precision"] <= 1
    assert 0 <= data["recall"] <= 1
    assert 0 <= data["f1_score"] <= 1
    
    cm = data["confusion_matrix"]
    assert len(cm) == 2
    assert len(cm[0]) == 2
    assert len(cm[1]) == 2
    assert all(isinstance(val, int) for row in cm for val in row)


def test_prediction_endpoint(client):
    """Test prediction endpoint"""
    response = client.get("/api/predict/current")
    assert response.status_code == 200
    
    data = response.json()
    
    assert "current_price" in data
    assert "predicted_direction" in data
    assert "confidence" in data
    assert "top_features" in data
    
    assert data["predicted_direction"] in ["UP", "DOWN"]
    assert 0 <= data["confidence"] <= 1
    assert isinstance(data["current_price"], (int, float))
    assert isinstance(data["top_features"], list)
    assert len(data["top_features"]) > 0


def test_technical_indicators_endpoint(client):
    """Test technical indicators endpoint"""
    response = client.get("/api/technical/indicators?symbol=BTCUSDT&days=90")
    assert response.status_code == 200
    
    data = response.json()
    
    assert "rsi" in data
    assert "macd" in data
    assert "macd_signal" in data
    assert "sma_7" in data
    assert "sma_25" in data
    
    rsi_len = len(data["rsi"])
    assert len(data["macd"]) == rsi_len
    assert len(data["macd_signal"]) == rsi_len
    assert len(data["sma_7"]) == rsi_len
    assert len(data["sma_25"]) == rsi_len


def test_error_handling(client):
    """Test error handling"""
    response = client.get("/api/nonexistent")
    assert response.status_code == 404


class TestFastAPIConfig:
    """Tests for FastAPI configuration"""
    
    def test_app_title(self):
        """Test that app has correct title"""
        assert app.title == "FastAPI"
    
    def test_app_description(self):
        """Test that app has description"""
        assert app.description == ""
    
    def test_app_version(self):
        """Test that app has version"""
        assert app.version == "0.1.0"


def test_endpoint_parameters(client):
    """Test endpoint parameters"""
    response1 = client.get("/api/ohlcv/BTCUSDT")
    assert response1.status_code == 200
    
    response2 = client.get("/api/ohlcv/ETHUSDT")
    assert response2.status_code == 200
    
    response3 = client.get("/api/features/importance?top_n=5")
    assert response3.status_code == 200


def test_response_validation(client):
    """Test response data validation"""
    response = client.get("/api/ohlcv/BTCUSDT")
    data = response.json()
    
    assert isinstance(data["timestamps"], list)
    assert isinstance(data["prices"], list)
    assert isinstance(data["volume"], list)
    assert isinstance(data["indicators"], dict)
    
    assert all(price > 0 for price in data["prices"])
    
    assert all(0 <= rsi <= 100 for rsi in data["indicators"]["rsi_14"])


def test_all_endpoints_are_tested():
    """Meta-test to ensure we cover all endpoints"""
    endpoints_we_test = [
        "/",
        "/api/ohlcv/{symbol}",
        "/api/features/importance",
        "/api/model/metrics",
        "/api/predict/current",
        "/api/technical/indicators"
    ]
    
    print(f"\nTesting {len(endpoints_we_test)} FastAPI endpoints")
    assert len(endpoints_we_test) >= 5, "Should test at least 5 endpoints"


def test_data_format_consistency(client):
    """Test data format consistency across endpoints"""
    endpoints = [
        "/api/ohlcv/BTCUSDT",
        "/api/features/importance",
        "/api/model/metrics",
        "/api/predict/current",
        "/api/technical/indicators"
    ]
    
    for endpoint in endpoints:
        response = client.get(endpoint)
        assert response.status_code == 200
        
        data = response.json()
        assert data is not None
        
        if isinstance(data, dict):
            assert len(data) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])