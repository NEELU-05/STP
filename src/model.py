"""
Model Module
Train and manage RandomForestClassifier for stock direction prediction.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
from typing import Tuple, Optional


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.
    Uses time-series aware split (no shuffle) to prevent data leakage.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector
    test_size : float, default=0.2
        Proportion of data to use for testing
    random_state : int, default=42
        Random state for reproducibility
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    print("✂️  Splitting data...")
    
    # For time-series data, we don't shuffle to maintain temporal order
    # Use the last test_size% of data for testing
    split_index = int(len(X) * (1 - test_size))
    
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Testing set: {len(X_test)} samples")
    print(f"   Train date range: {X_train.index[0].date()} to {X_train.index[-1].date()}")
    print(f"   Test date range: {X_test.index[0].date()} to {X_test.index[-1].date()}")
    
    return X_train, X_test, y_train, y_test


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_split: int = 5,
    random_state: int = 42
) -> RandomForestClassifier:
    """
    Train a RandomForestClassifier model.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix
    y_train : pd.Series
        Training target vector
    n_estimators : int, default=100
        Number of trees in the forest
    max_depth : int, default=10
        Maximum depth of trees
    min_samples_split : int, default=5
        Minimum samples required to split a node
    random_state : int, default=42
        Random state for reproducibility
    
    Returns
    -------
    RandomForestClassifier
        Trained model
    """
    print("\n🌲 Training Random Forest Classifier...")
    print(f"   Parameters:")
    print(f"   - n_estimators: {n_estimators}")
    print(f"   - max_depth: {max_depth}")
    print(f"   - min_samples_split: {min_samples_split}")
    
    # Initialize model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1,  # Use all CPU cores
        verbose=0
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    print("   ✅ Model training complete!")
    
    # Show feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n📊 Top 5 Important Features:")
    for idx, row in feature_importance.head(5).iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    return model


def save_model(model: RandomForestClassifier, filepath: str) -> None:
    """
    Save trained model to disk.
    
    Parameters
    ----------
    model : RandomForestClassifier
        Trained model to save
    filepath : str
        Path where model will be saved
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save model
    joblib.dump(model, filepath)
    print(f"\n💾 Model saved to: {filepath}")


def load_model(filepath: str) -> Optional[RandomForestClassifier]:
    """
    Load a trained model from disk.
    
    Parameters
    ----------
    filepath : str
        Path to saved model file
    
    Returns
    -------
    RandomForestClassifier or None
        Loaded model, or None if loading fails
    """
    try:
        if not os.path.exists(filepath):
            print(f"❌ Model file not found: {filepath}")
            return None
        
        model = joblib.load(filepath)
        print(f"✅ Model loaded from: {filepath}")
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None


def predict(model: RandomForestClassifier, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions using trained model.
    
    Parameters
    ----------
    model : RandomForestClassifier
        Trained model
    X : pd.DataFrame
        Feature matrix for prediction
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Predictions (0/1) and prediction probabilities
    """
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    return predictions, probabilities


def get_feature_importance(model: RandomForestClassifier, feature_names: list) -> pd.DataFrame:
    """
    Get feature importance from trained model.
    
    Parameters
    ----------
    model : RandomForestClassifier
        Trained model
    feature_names : list
        List of feature names
    
    Returns
    -------
    pd.DataFrame
        DataFrame with features and their importance scores
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance_df


if __name__ == "__main__":
    # Example usage
    print("Model Training - Example Usage\n")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)],
        index=dates
    )
    y = pd.Series(np.random.randint(0, 2, n_samples), index=dates)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Make predictions
    predictions, probabilities = predict(model, X_test)
    print(f"\n🎯 Sample predictions: {predictions[:10]}")
    print(f"📊 Prediction probabilities shape: {probabilities.shape}")
