import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import os

class MLService:
    def __init__(self):
        self.model = None
        self.features = None
        self.feature_importance = None
        self.load_model()
    
    def load_model(self):
        """Загрузка обученной модели"""
        try:
            if os.path.exists("../improved_catboost_model.pkl"):
                model_data = joblib.load("../improved_catboost_model.pkl")
                self.model = model_data["model"]
                self.features = model_data["features"]
                self.feature_importance = model_data["feature_importance"]
                print("✅ ML модель загружена")
            else:
                print("❌ Модель не найдена, запустите train_model.py")
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
    
    def get_feature_importance(self, top_n: int = 20):
        """Получение важности фичей"""
        if self.feature_importance is not None:
            return self.feature_importance.head(top_n).to_dict('records')
        return []
    
    def get_model_metrics(self, df):
        """Расчет метрик модели на последних данных"""
        if self.model is None or self.features is None:
            return None
        
        # Подготовка данных для предсказания
        X = df[self.features].fillna(0)
        y = df['target_direction']
        
        # Временной сплит
        split_idx = int(len(df) * 0.8)
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]
        
        # Предсказания
        y_pred = self.model.predict(X_test)
        
        # Метрики
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        return {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm
        }
    
    def predict_current(self, df):
        """Предсказание для последней доступной точки"""
        if self.model is None or self.features is None:
            return None
        
        # Берем последнюю точку
        latest = df.iloc[-1]
        X_latest = latest[self.features].fillna(0).values.reshape(1, -1)
        
        # Предсказание
        prediction = self.model.predict(X_latest)[0]
        probability = self.model.predict_proba(X_latest)[0]
        confidence = max(probability)
        
        direction = "UP" if prediction == 1 else "DOWN"
        current_price = latest['close']
        
        return {
            "current_price": current_price,
            "predicted_direction": direction,
            "confidence": confidence,
            "top_features": self.get_feature_importance(10)
        }

# Глобальный экземпляр
ml_service = MLService()