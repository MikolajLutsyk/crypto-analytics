from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, List, Optional

app = FastAPI(title="BTC Prediction API")

# Глобальная переменная для хранения модели
model_data = None

class Features(BaseModel):
    data: Dict[str, float]

class PredictionResponse(BaseModel):
    prediction: int
    probability_up: float
    probability_down: float
    confidence: float

def load_latest_model():
    """Загрузка самой свежей модели из текущей директории"""
    global model_data
    
    try:
        # Ищем все файлы моделей
        model_files = [f for f in os.listdir('.') if f.startswith('model') and f.endswith('.pkl')]
        
        if not model_files:
            print("❌ Файлы моделей не найдены в текущей директории")
            print("📁 Текущая директория:", os.getcwd())
            print("📁 Содержимое директории:", os.listdir('.'))
            return False
        
        # Выбираем самую свежую модель
        latest_model = sorted(model_files)[-1]
        print(f"🔍 Найдена модель: {latest_model}")
        
        # Загружаем модель
        with open(latest_model, 'rb') as f:
            model_data = pickle.load(f)
        
        print(f"✅ Модель успешно загружена:")
        print(f"   - Тип: {type(model_data['model'])}")
        print(f"   - Accuracy: {model_data.get('accuracy', 'N/A')}")
        print(f"   - Фич: {len(model_data['features'])}")
        print(f"   - Время: {model_data.get('timestamp', 'N/A')}")
        
        # Показываем первые 10 фич
        print(f"   - Пример фич: {model_data['features'][:10]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.on_event("startup")
async def startup_event():
    """Загрузка модели при старте приложения"""
    print("🚀 Запуск приложения...")
    success = load_latest_model()
    if not success:
        print("❌ МОДЕЛЬ НЕ ЗАГРУЖЕНА! API будет возвращать ошибки.")

@app.get("/")
async def root():
    return {"message": "BTC Prediction API", "model_loaded": model_data is not None}

@app.get("/model_info")
async def get_model_info():
    """Информация о загруженной модели"""
    if not model_data:
        raise HTTPException(status_code=500, detail="Модель не загружена")
    
    return {
        "model_loaded": True,
        "features_count": len(model_data['features']),
        "features": model_data['features'],
        "accuracy": model_data.get('accuracy', 'N/A'),
        "timestamp": model_data.get('timestamp', 'N/A'),
        "model_type": str(type(model_data['model']))
    }

@app.get("/health")
async def health_check():
    """Проверка статуса API и модели"""
    return {
        "status": "ok",
        "model_loaded": model_data is not None,
        "features_count": len(model_data['features']) if model_data else 0
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: Features):
    """Предсказание направления цены BTC"""
    if not model_data:
        raise HTTPException(status_code=500, detail="Модель не загружена. Проверьте логи сервера.")
    
    try:
        # Создаем DataFrame с правильным порядком фич
        input_data = []
        missing_features = []
        
        for feature in model_data['features']:
            if feature in features.data:
                input_data.append(features.data[feature])
            else:
                missing_features.append(feature)
                input_data.append(0.0)  # Заполняем нулем если фича отсутствует
        
        if missing_features:
            print(f"⚠️ Отсутствующие фичи: {missing_features}")
        
        # Преобразуем в DataFrame
        X = pd.DataFrame([input_data], columns=model_data['features'])
        
        # Масштабирование если scaler есть
        if 'scaler' in model_data and model_data['scaler'] is not None:
            X = model_data['scaler'].transform(X)
        
        # Предсказание
        model = model_data['model']
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        return {
            "prediction": int(prediction),
            "probability_up": float(probabilities[1]),
            "probability_down": float(probabilities[0]),
            "confidence": float(max(probabilities))
        }
        
    except Exception as e:
        print(f"❌ Ошибка предсказания: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)