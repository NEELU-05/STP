"""
Visualization Module
Create visualizations for predictions and model performance.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional
import os


# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_predictions(
    dates: pd.DatetimeIndex,
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    symbol: str = "Stock",
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot actual vs predicted stock direction over time.
    
    Parameters
    ----------
    dates : pd.DatetimeIndex
        Dates for x-axis
    y_actual : np.ndarray
        Actual direction values (0/1)
    y_pred : np.ndarray
        Predicted direction values (0/1)
    symbol : str, default="Stock"
        Stock symbol for title
    save_path : str, optional
        Path to save the plot
    show : bool, default=True
        Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot actual and predicted as step plots
    ax.step(dates, y_actual, where='post', label='Actual Direction', 
            color='#10b981', linewidth=2, alpha=0.7)
    ax.step(dates, y_pred, where='post', label='Predicted Direction', 
            color='#6366f1', linewidth=2, alpha=0.7, linestyle='--')
    
    # Customize plot
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Direction (0=Down, 1=Up)', fontsize=12, fontweight='bold')
    ax.set_title(f'{symbol} - Actual vs Predicted Direction', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Saved plot to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot confusion matrix as a heatmap.
    
    Parameters
    ----------
    conf_matrix : np.ndarray
        Confusion matrix (2x2)
    save_path : str, optional
        Path to save the plot
    show : bool, default=True
        Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar_kws={'label': 'Count'},
        xticklabels=['Down (0)', 'Up (1)'],
        yticklabels=['Down (0)', 'Up (1)'],
        ax=ax,
        linewidths=2,
        linecolor='white',
        square=True
    )
    
    # Customize plot
    ax.set_xlabel('Predicted Direction', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual Direction', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
    
    # Add accuracy annotation
    accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / conf_matrix.sum()
    plt.text(1, -0.3, f'Accuracy: {accuracy:.2%}', 
             ha='center', fontsize=11, fontweight='bold',
             transform=ax.transAxes)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Saved plot to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_feature_importance(
    model,
    feature_names: list,
    top_n: int = 10,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot feature importance as a horizontal bar chart.
    
    Parameters
    ----------
    model : RandomForestClassifier
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    top_n : int, default=10
        Number of top features to display
    save_path : str, optional
        Path to save the plot
    show : bool, default=True
        Whether to display the plot
    """
    # Get feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True).tail(top_n)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create horizontal bar chart
    bars = ax.barh(importance_df['feature'], importance_df['importance'], 
                   color='#8b5cf6', alpha=0.8)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, importance_df['importance'])):
        ax.text(value, bar.get_y() + bar.get_height()/2, 
                f'{value:.4f}', 
                ha='left', va='center', fontsize=9, 
                fontweight='bold', color='#1f2937')
    
    # Customize plot
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Features', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Most Important Features', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Saved plot to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_price_with_predictions(
    data: pd.DataFrame,
    predictions: np.ndarray,
    symbol: str = "Stock",
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot stock price with prediction markers.
    
    Parameters
    ----------
    data : pd.DataFrame
        Stock data with 'Close' column
    predictions : np.ndarray
        Predicted directions (0/1)
    symbol : str, default="Stock"
        Stock symbol for title
    save_path : str, optional
        Path to save the plot
    show : bool, default=True
        Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot closing price
    ax.plot(data.index, data['Close'], label='Close Price', 
            color='#1f2937', linewidth=2, alpha=0.7)
    
    # Add prediction markers
    up_mask = predictions == 1
    down_mask = predictions == 0
    
    ax.scatter(data.index[up_mask], data['Close'].values[up_mask], 
               color='#10b981', marker='^', s=50, alpha=0.6, 
               label='Predicted Up', zorder=5)
    ax.scatter(data.index[down_mask], data['Close'].values[down_mask], 
               color='#ef4444', marker='v', s=50, alpha=0.6, 
               label='Predicted Down', zorder=5)
    
    # Customize plot
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
    ax.set_title(f'{symbol} - Price with Direction Predictions', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Saved plot to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def create_all_visualizations(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    conf_matrix: np.ndarray,
    symbol: str = "Stock",
    output_dir: str = "data/plots"
) -> None:
    """
    Create and save all visualizations.
    
    Parameters
    ----------
    model : RandomForestClassifier
        Trained model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Actual test targets
    y_pred : np.ndarray
        Predicted targets
    conf_matrix : np.ndarray
        Confusion matrix
    symbol : str, default="Stock"
        Stock symbol
    output_dir : str, default="data/plots"
        Directory to save plots
    """
    print("\n📊 Creating visualizations...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Actual vs Predicted
    plot_predictions(
        X_test.index,
        y_test.values,
        y_pred,
        symbol=symbol,
        save_path=f"{output_dir}/{symbol}_predictions.png",
        show=False
    )
    
    # 2. Confusion Matrix
    plot_confusion_matrix(
        conf_matrix,
        save_path=f"{output_dir}/{symbol}_confusion_matrix.png",
        show=False
    )
    
    # 3. Feature Importance
    plot_feature_importance(
        model,
        X_test.columns.tolist(),
        top_n=10,
        save_path=f"{output_dir}/{symbol}_feature_importance.png",
        show=False
    )
    
    print(f"   ✅ All visualizations saved to: {output_dir}/")


if __name__ == "__main__":
    # Example usage
    print("Visualization Module - Example Usage\n")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
    y_actual = np.random.randint(0, 2, n_samples)
    y_pred = y_actual.copy()
    # Add some errors
    error_indices = np.random.choice(n_samples, size=20, replace=False)
    y_pred[error_indices] = 1 - y_pred[error_indices]
    
    # Plot predictions
    plot_predictions(dates, y_actual, y_pred, symbol="AAPL", show=True)
    
    # Create confusion matrix
    from sklearn.metrics import confusion_matrix
    conf_matrix = confusion_matrix(y_actual, y_pred)
    plot_confusion_matrix(conf_matrix, show=True)
