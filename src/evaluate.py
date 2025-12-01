"""
Evaluation Module
Evaluate model performance with comprehensive metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)
from typing import Dict, Tuple


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict:
    """
    Evaluate model performance and return all metrics.
    
    Parameters
    ----------
    model : RandomForestClassifier
        Trained model
    X_test : pd.DataFrame
        Test feature matrix
    y_test : pd.Series
        Test target vector
    
    Returns
    -------
    Dict
        Dictionary containing all evaluation metrics
    """
    print("\n📊 Evaluating model performance...\n")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Get classification report
    class_report = classification_report(
        y_test,
        y_pred,
        target_names=['Down (0)', 'Up (1)'],
        output_dict=True
    )
    
    # Package results
    results = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': class_report,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'actual': y_test
    }
    
    return results


def print_metrics(results: Dict) -> None:
    """
    Print evaluation metrics in a formatted way.
    
    Parameters
    ----------
    results : Dict
        Dictionary containing evaluation metrics from evaluate_model()
    """
    print("=" * 60)
    print("📈 MODEL PERFORMANCE METRICS")
    print("=" * 60)
    
    # Overall accuracy
    print(f"\n🎯 Overall Accuracy: {results['accuracy']:.2%}")
    
    # Confusion Matrix
    print(f"\n📊 Confusion Matrix:")
    print("-" * 40)
    conf_matrix = results['confusion_matrix']
    print(f"                 Predicted Down  Predicted Up")
    print(f"Actual Down           {conf_matrix[0][0]:>4}          {conf_matrix[0][1]:>4}")
    print(f"Actual Up             {conf_matrix[1][0]:>4}          {conf_matrix[1][1]:>4}")
    
    # Calculate confusion matrix percentages
    tn, fp, fn, tp = conf_matrix.ravel()
    total = tn + fp + fn + tp
    
    print(f"\n📉 Breakdown:")
    print(f"   True Negatives (Correctly predicted Down):  {tn} ({tn/total:.1%})")
    print(f"   True Positives (Correctly predicted Up):    {tp} ({tp/total:.1%})")
    print(f"   False Positives (Incorrectly predicted Up): {fp} ({fp/total:.1%})")
    print(f"   False Negatives (Incorrectly predicted Down): {fn} ({fn/total:.1%})")
    
    # Precision, Recall, F1-Score
    print(f"\n📏 Detailed Metrics:")
    print("-" * 40)
    print(f"   Precision: {results['precision']:.2%}")
    print(f"   Recall:    {results['recall']:.2%}")
    print(f"   F1-Score:  {results['f1_score']:.2%}")
    
    # Per-class metrics
    print(f"\n📋 Per-Class Performance:")
    print("-" * 40)
    class_report = results['classification_report']
    
    for class_name in ['Down (0)', 'Up (1)']:
        metrics = class_report[class_name]
        print(f"\n   {class_name}:")
        print(f"      Precision: {metrics['precision']:.2%}")
        print(f"      Recall:    {metrics['recall']:.2%}")
        print(f"      F1-Score:  {metrics['f1-score']:.2%}")
        print(f"      Support:   {int(metrics['support'])} samples")
    
    # Prediction distribution
    predictions = results['predictions']
    actual = results['actual']
    
    print(f"\n📊 Prediction Distribution:")
    print("-" * 40)
    print(f"   Predicted Up:   {(predictions == 1).sum()} ({(predictions == 1).sum()/len(predictions):.1%})")
    print(f"   Predicted Down: {(predictions == 0).sum()} ({(predictions == 0).sum()/len(predictions):.1%})")
    print(f"   Actual Up:      {(actual == 1).sum()} ({(actual == 1).sum()/len(actual):.1%})")
    print(f"   Actual Down:    {(actual == 0).sum()} ({(actual == 0).sum()/len(actual):.1%})")
    
    print("\n" + "=" * 60)


def get_prediction_confidence(probabilities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate prediction confidence scores.
    
    Parameters
    ----------
    probabilities : np.ndarray
        Prediction probabilities from model.predict_proba()
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Confidence scores and predicted classes
    """
    # Confidence is the maximum probability for each prediction
    confidence = np.max(probabilities, axis=1)
    predicted_class = np.argmax(probabilities, axis=1)
    
    return confidence, predicted_class


def analyze_errors(
    y_test: pd.Series,
    y_pred: np.ndarray,
    X_test: pd.DataFrame,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Analyze prediction errors to identify patterns.
    
    Parameters
    ----------
    y_test : pd.Series
        Actual target values
    y_pred : np.ndarray
        Predicted values
    X_test : pd.DataFrame
        Test features
    top_n : int, default=10
        Number of top errors to return
    
    Returns
    -------
    pd.DataFrame
        DataFrame with error analysis
    """
    # Create error dataframe
    errors_df = pd.DataFrame({
        'actual': y_test.values,
        'predicted': y_pred,
        'error': (y_test.values != y_pred).astype(int),
        'date': y_test.index
    })
    
    # Filter only errors
    errors_only = errors_df[errors_df['error'] == 1].copy()
    
    if len(errors_only) == 0:
        print("🎉 No prediction errors found!")
        return pd.DataFrame()
    
    print(f"\n❌ Error Analysis:")
    print(f"   Total errors: {len(errors_only)} out of {len(y_test)} ({len(errors_only)/len(y_test):.1%})")
    
    # Analyze error types
    false_positives = len(errors_only[(errors_only['actual'] == 0) & (errors_only['predicted'] == 1)])
    false_negatives = len(errors_only[(errors_only['actual'] == 1) & (errors_only['predicted'] == 0)])
    
    print(f"   False Positives (predicted Up, actually Down): {false_positives}")
    print(f"   False Negatives (predicted Down, actually Up): {false_negatives}")
    
    return errors_only.head(top_n)


if __name__ == "__main__":
    # Example usage
    print("Model Evaluation - Example Usage\n")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 200
    
    # Simulate predictions
    y_test = pd.Series(np.random.randint(0, 2, n_samples))
    y_pred = y_test.copy()
    # Add some errors
    error_indices = np.random.choice(n_samples, size=40, replace=False)
    y_pred.iloc[error_indices] = 1 - y_pred.iloc[error_indices]
    
    # Create mock model with predict method
    class MockModel:
        def predict(self, X):
            return y_pred.values
        
        def predict_proba(self, X):
            proba = np.random.rand(len(X), 2)
            proba = proba / proba.sum(axis=1, keepdims=True)
            return proba
    
    model = MockModel()
    X_test = pd.DataFrame(np.random.randn(n_samples, 5))
    
    # Evaluate
    results = evaluate_model(model, X_test, y_test)
    print_metrics(results)
