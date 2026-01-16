import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_feature_data():
    """Load data from features.csv"""
    try:
        df = pd.read_csv("data/features.csv", index_col="open_time", parse_dates=True)
        print(f"Loaded {len(df)} rows from data/features.csv")
        return df
    except FileNotFoundError:
        print("File data/features.csv not found. Run feature_engineering.py first")
        return None

def prepare_feature_data(df):
    exclude_cols = ['close_future', 'future_return', 'target_direction', 'target_3class']
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    X = df[feature_cols].fillna(0)
    y = df['target_direction']
    
    print(f"Using {len(feature_cols)} features")
    return X, y, feature_cols

def select_best_features(X, y, k=30):
    selector = SelectKBest(f_classif, k=min(k, X.shape[1]))
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    
    print(f"Selected {len(selected_features)} best features")
    return selected_features

def create_time_based_split(df, test_size=0.2):
    split_idx = int(len(df) * (1 - test_size))
    train_mask = df.index <= df.index[split_idx]
    test_mask = df.index > df.index[split_idx]
    
    return train_mask, test_mask

def prepare_categorical_features(selected_features):
    cat_features_indices = []
    cat_features_names = []
    
    true_categorical = ['hour', 'day_of_week', 'day_of_month', 'is_weekend']
    
    for i, col in enumerate(selected_features):
        if col in true_categorical:
            cat_features_indices.append(i)
            cat_features_names.append(col)
    
    print(f"Categorical features ({len(cat_features_indices)}): {cat_features_names}")
    return cat_features_indices

def train_model(X, y, selected_features):
    train_mask, test_mask = create_time_based_split(X)
    
    X_train = X[selected_features].loc[train_mask]
    X_test = X[selected_features].loc[test_mask]
    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]
    
    print(f"📈 Training set size: {len(X_train)}")
    print(f"📊 Test set size: {len(X_test)}")
    print(f"📊 Class balance in training set:")
    print(y_train.value_counts().sort_index())
    
    cat_features_indices = prepare_categorical_features(selected_features)
    
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
        auto_class_weights='Balanced'
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        cat_features=cat_features_indices,
        use_best_model=True
    )
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n🎯 Accuracy: {accuracy:.3f}")
    print(f"⚖️  Balanced Accuracy: {balanced_acc:.3f}")
    print(f"📊 Precision: {precision:.3f}")
    print(f"📊 Recall: {recall:.3f}")
    print(f"📊 F1-score: {f1:.3f}")
    
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["DOWN", "UP"]))
    
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["DOWN", "UP"], yticklabels=["DOWN", "UP"])
    plt.title("CatBoost Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    feature_importance = model.get_feature_importance()
    feature_importance_df = pd.DataFrame({
        'feature': selected_features,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance_df.head(20), x='importance', y='feature')
    plt.title('Top 20 Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, accuracy, X_test, y_test, y_pred, feature_importance_df

def save_model(model, features, accuracy, feature_importance):
    """Save model and metadata"""
    model_data = {
        "model": model,
        "features": features,
        "accuracy": accuracy,
        "feature_importance": feature_importance.to_dict('records'),
        "timestamp": pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    }
    
    with open("catboost_model.pkl", "wb") as f:
        pickle.dump(model_data, f)
    
    print("💾 Model saved → catboost_model.pkl")

def run_pipeline():
    """Main training pipeline"""
    print("🚀 Starting CatBoost training pipeline...")
    
    # 1. Load data
    df = load_feature_data()
    if df is None:
        return
    
    # 2. Prepare data
    X, y, all_features = prepare_feature_data(df)
    
    # 3. Feature selection
    selected_features = select_best_features(X, y, k=40)
    
    # 4. Train model
    model, accuracy, X_test, y_test, y_pred, feature_importance = train_model(X, y, selected_features)
    
    # 5. Save model
    save_model(model, selected_features, accuracy, feature_importance)
    
    # 6. Evaluate results
    if accuracy >= 0.6:
        print(f"🎉 Excellent result! Accuracy: {accuracy:.2%}")
        print("✅ Model reached target accuracy of 60%+")
    elif accuracy >= 0.55:
        print(f"⚠️  Acceptable result: {accuracy:.2%}")
        print("ℹ️  You can try improving it via hyperparameter tuning")
    else:
        print(f"❌ Low accuracy: {accuracy:.2%}")
        print("💡 Recommendation: increase dataset size or add more features")
    
    # Print top-10 features
    print("\n🏆 Top-10 most important features:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"  {i+1:2d}. {row['feature']}: {row['importance']:.4f}")

if __name__ == "__main__":
    run_pipeline()
