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
        """Loading trained model"""
        try:
            possible_paths = [
                "catboost_model.pkl",
                "./catboost_model.pkl",
                "../catboost_model.pkl",
                "../../catboost_model.pkl"
            ]
            
            loaded = False
            for model_path in possible_paths:
                if os.path.exists(model_path):
                    print(f"✅ Найдена модель: {model_path}")
                    model_data = joblib.load(model_path)
                    self.model = model_data["model"]
                    self.features = model_data["features"]
                    
                    if "feature_importance" in model_data:
                        self.feature_importance = pd.DataFrame(model_data["feature_importance"])
                    elif "feature_names" in model_data:
                        self.feature_importance = pd.DataFrame({
                            'feature': model_data["feature_names"],
                            'importance': model_data["importance_values"]
                        })
                    else:
                        print("⚠️  feature_importance was saved in an unknown format")
                        self.feature_importance = pd.DataFrame()
                    
                    loaded = True
                    print("SUCCESS -- ML model loaded")
                    print(f"📊 Количество признаков: {len(self.features)}")
                    if not self.feature_importance.empty:
                        print(f"📊 Feature importance loaded: {len(self.feature_importance)} lines")
                    break
            
            if not loaded:
                print("FAILURE -- model not found in any path, run train_model.py first")
                
        except Exception as e:
            print(f"ECXEPTION -- failed to load model: {e}")
            import traceback
            print(traceback.format_exc())
    
    def get_feature_importance(self, top_n: int = 20):
        """Getting feature importance"""
        if self.feature_importance is not None:
            return self.feature_importance.head(top_n).to_dict('records')
        return []
    
    def get_model_metrics(self, df):
        """Determining metrics based on latest data"""
        if self.model is None or self.features is None:
            return None
        
        # Getting data ready for prediction
        X = df[self.features].fillna(0)
        y = df['target_direction']
        
        # Time split
        split_idx = int(len(df) * 0.8)
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]
        
        # Prediction
        y_pred = self.model.predict(X_test)
        
        # Metrics
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
        print(f"\n===== PREDICT_CURRENT STARTED =====")
        
        if self.model is None or self.features is None:
            print("Model or features not loaded")
            return None
        
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            print(f"Features missing: {missing[:5]}...")
            return None
        try:
            latest = df.iloc[-1]
            X_latest_df = pd.DataFrame([latest[self.features]])
            X_latest_df = X_latest_df.fillna(0)
            
            prediction = self.model.predict(X_latest_df)[0]
            probability = self.model.predict_proba(X_latest_df)[0]
            confidence = max(probability)
            
            print(f"Prediction: {prediction}, confidence: {confidence}")
            
            return {
                "current_price": float(latest['close']),
                "predicted_direction": "UP" if prediction == 1 else "DOWN",
                "confidence": float(confidence),
                "top_features": self.get_feature_importance(10)
            }
        except Exception as e:
            print(f"Critical error in predict_current: {e}")
            import traceback
            print(traceback.format_exc())
            return None

# Global object
ml_service = MLService()