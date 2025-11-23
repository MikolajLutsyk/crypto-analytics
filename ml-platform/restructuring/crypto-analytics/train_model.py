import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_feature_data():
    """Загрузка данных из features.csv"""
    try:
        df = pd.read_csv("data/features.csv", index_col="open_time", parse_dates=True)
        print(f"✅ Загружено {len(df)} строк из data/features.csv")
        return df
    except FileNotFoundError:
        print("❌ Файл data/features.csv не найден. Сначала запустите feature_engineering.py")
        return None

def prepare_feature_data(df):
    """Подготовка данных для обучения"""
    # Удаляем целевые переменные и ненужные колонки
    exclude_cols = ['close_future', 'future_return', 'target_direction', 'target_3class']
    
    # Оставляем только числовые колонки
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    X = df[feature_cols].fillna(0)
    y = df['target_direction']
    
    print(f"📊 Используется {len(feature_cols)} признаков")
    return X, y, feature_cols

def select_best_features(X, y, k=30):
    """Выбор лучших признаков"""
    selector = SelectKBest(f_classif, k=min(k, X.shape[1]))
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    
    print(f"🎯 Выбрано {len(selected_features)} лучших признаков")
    return selected_features

def create_time_based_split(df, test_size=0.2):
    """Создание временного сплита (важно для временных рядов)"""
    split_idx = int(len(df) * (1 - test_size))
    train_mask = df.index <= df.index[split_idx]
    test_mask = df.index > df.index[split_idx]
    
    return train_mask, test_mask

def prepare_categorical_features(selected_features):
    """Подготовка категориальных признаков - ТОЛЬКО настоящие категориальные"""
    cat_features_indices = []
    cat_features_names = []
    
    # ТОЛЬКО настоящие категориальные признаки
    true_categorical = ['hour', 'day_of_week', 'day_of_month', 'is_weekend']
    
    for i, col in enumerate(selected_features):
        if col in true_categorical:
            cat_features_indices.append(i)
            cat_features_names.append(col)
    
    print(f"🏷️ Категориальные признаки ({len(cat_features_indices)}): {cat_features_names}")
    return cat_features_indices

def train_model(X, y, selected_features):
    """Обучение модели CatBoost с балансировкой классов"""
    # Временной сплит
    train_mask, test_mask = create_time_based_split(X)
    
    X_train = X[selected_features].loc[train_mask]
    X_test = X[selected_features].loc[test_mask]
    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]
    
    print(f"📈 Размер обучающей выборки: {len(X_train)}")
    print(f"📊 Размер тестовой выборки: {len(X_test)}")
    print(f"📊 Баланс классов в обучающей выборке:")
    print(y_train.value_counts().sort_index())
    
    # БАЛАНСИРОВКА КЛАССОВ С ПОМОЩЬЮ SMOTE
    try:
        from imblearn.over_sampling import SMOTE
        print("🔄 Применяем SMOTE для балансировки классов...")
        
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        print(f"📈 Размер сбалансированной обучающей выборки: {len(X_train_balanced)}")
        print(f"📊 Баланс классов после SMOTE:")
        print(pd.Series(y_train_balanced).value_counts().sort_index())
        
    except ImportError:
        print("⚠️  imblearn не установлен, используем встроенную балансировку CatBoost")
        X_train_balanced, y_train_balanced = X_train, y_train
    
    # Получаем индексы категориальных признаков
    cat_features_indices = prepare_categorical_features(selected_features)
    
    # Обучение модели - ИСПОЛЬЗУЕМ СБАЛАНСИРОВАННЫЕ ДАННЫЕ
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        random_strength=0.5,
        bagging_temperature=0.8,
        od_type='Iter',
        od_wait=50,
        loss_function='Logloss',
        eval_metric='Accuracy',
        random_seed=42,
        verbose=100,
        auto_class_weights='Balanced'  # Дополнительная балансировка
    )
    
    # Обучение на СБАЛАНСИРОВАННЫХ данных
    model.fit(
        X_train_balanced, y_train_balanced,
        eval_set=(X_test, y_test),
        cat_features=cat_features_indices,
        use_best_model=True
    )
    
    # Предсказания и метрики
    y_pred = model.predict(X_test)
    
    # Дополнительные метрики для несбалансированных данных
    from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n🎯 Точность (Accuracy): {accuracy:.3f}")
    print(f"⚖️  Сбалансированная точность: {balanced_acc:.3f}")
    print(f"📊 Precision: {precision:.3f}")
    print(f"📊 Recall: {recall:.3f}")
    print(f"📊 F1-score: {f1:.3f}")
    
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["DOWN", "UP"]))
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["DOWN", "UP"], yticklabels=["DOWN", "UP"])
    plt.title("Confusion Matrix (после балансировки)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig('confusion_matrix_balanced.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature Importance
    feature_importance = model.get_feature_importance()
    feature_importance_df = pd.DataFrame({
        'feature': selected_features,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance_df.head(20), x='importance', y='feature')
    plt.title('Top 20 Feature Importance (после балансировки)')
    plt.tight_layout()
    plt.savefig('feature_importance_balanced.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, accuracy, X_test, y_test, y_pred, feature_importance_df

def save_model(model, features, accuracy, feature_importance):
    """Сохранение модели и метаданных"""
    model_data = {
        "model": model,
        "features": features,
        "accuracy": accuracy,
        "feature_importance": feature_importance,
        "timestamp": pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    }
    
    with open("improved_catboost_model.pkl", "wb") as f:
        pickle.dump(model_data, f)
    
    print("💾 Модель сохранена → improved_catboost_model.pkl")

def run_improved_pipeline():
    """Основной пайплайн обучения"""
    print("🚀 Запуск улучшенного пайплайна обучения...")
    
    # 1. Загрузка данных
    df = load_feature_data()
    if df is None:
        return
    
    # 2. Подготовка данных
    X, y, all_features = prepare_feature_data(df)
    
    # 3. Выбор признаков
    selected_features = select_best_features(X, y, k=40)
    
    # 4. Обучение модели
    model, accuracy, X_test, y_test, y_pred, feature_importance = train_model(X, y, selected_features)
    
    # 5. Сохранение модели
    save_model(model, selected_features, accuracy, feature_importance)
    
    # 6. Оценка результатов
    if accuracy >= 0.6:
        print(f"🎉 Отличный результат! Точность: {accuracy:.2%}")
        print("✅ Модель достигла целевой точности 60%+")
    elif accuracy >= 0.55:
        print(f"⚠️  Приемлемый результат: {accuracy:.2%}")
        print("ℹ️  Можно попробовать улучшить через настройку гиперпараметров")
    else:
        print(f"❌ Низкая точность: {accuracy:.2%}")
        print("💡 Рекомендации: попробуйте увеличить объем данных или добавить дополнительные признаки")
    
    # Вывод топ-10 признаков
    print("\n🏆 Топ-10 самых важных признаков:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"  {i+1:2d}. {row['feature']}: {row['importance']:.4f}")

if __name__ == "__main__":
    run_improved_pipeline()