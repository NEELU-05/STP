"""
Stock Movement Direction Predictor - Main Notebook
This notebook demonstrates the complete workflow from data loading to prediction.

To convert this to Jupyter notebook, run:
    jupyter nbconvert --to notebook --execute main_notebook.py
Or simply copy the cells into a new Jupyter notebook.
"""

# %% [markdown]
# # Stock Movement Direction Predictor
# 
# ## Complete End-to-End Workflow
# 
# This notebook demonstrates how to:
# 1. Download stock data from Yahoo Finance
# 2. Engineer technical indicators
# 3. Train a Random Forest model
# 4. Evaluate model performance
# 5. Visualize results
#
# **Target**: Predict whether stock price will go Up (1) or Down (0) the next day

# %% [markdown]
# ## 1. Setup and Imports

# %%
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import download_stock_data, get_available_symbols
from src.feature_engineering import prepare_features
from src.model import train_model, split_data, get_feature_importance
from src.evaluate import evaluate_model, print_metrics
from src.visualize import (
    plot_predictions, 
    plot_confusion_matrix, 
    plot_feature_importance,
    create_all_visualizations
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("✅ All modules imported successfully!")

# %% [markdown]
# ## 2. View Available Stock Symbols

# %%
available_stocks = get_available_symbols()
print("📋 Available Stock Symbols:")
print(", ".join(available_stocks))

# %% [markdown]
# ## 3. Download Stock Data
# 
# We'll use Apple (AAPL) as an example. The function downloads all available historical data.

# %%
# Select stock symbol
SYMBOL = "AAPL"

print(f"\n{'='*60}")
print(f"Downloading data for {SYMBOL}...")
print(f"{'='*60}\n")

# Download data
stock_data = download_stock_data(SYMBOL)

# Display basic information
if stock_data is not None:
    print(f"\n📊 Data Overview:")
    print(f"   Shape: {stock_data.shape}")
    print(f"   Columns: {list(stock_data.columns)}")
    print(f"   Date Range: {stock_data.index[0].date()} to {stock_data.index[-1].date()}")
    print(f"\n📈 First 5 rows:")
    print(stock_data.head())
    print(f"\n📉 Last 5 rows:")
    print(stock_data.tail())

# %% [markdown]
# ## 4. Feature Engineering
# 
# Create technical indicators:
# - Simple Moving Averages (SMA): 10, 20, 50 days
# - Exponential Moving Averages (EMA): 12, 26 days
# - Relative Strength Index (RSI): 14 days
# - Momentum: 10 days
# - Daily Returns

# %%
print(f"\n{'='*60}")
print("Creating Features and Target Variable...")
print(f"{'='*60}\n")

# Prepare features
X, y = prepare_features(stock_data)

print(f"\n✅ Feature Engineering Complete!")
print(f"\n📊 Feature Matrix (X):")
print(f"   Shape: {X.shape}")
print(f"   Features: {list(X.columns)}")
print(f"\n🎯 Target Variable (y):")
print(f"   Shape: {y.shape}")
print(f"   Distribution:")
print(f"   - Up (1): {(y == 1).sum()} ({(y == 1).sum()/len(y):.1%})")
print(f"   - Down (0): {(y == 0).sum()} ({(y == 0).sum()/len(y):.1%})")

# Display sample features
print(f"\n📋 Sample Features:")
print(X.head())

# %% [markdown]
# ## 5. Split Data into Training and Testing Sets
# 
# We use a time-series aware split (no shuffling) to prevent data leakage.
# - Training: 80% (earlier data)
# - Testing: 20% (recent data)

# %%
print(f"\n{'='*60}")
print("Splitting Data...")
print(f"{'='*60}\n")

X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

# %% [markdown]
# ## 6. Train Random Forest Model
# 
# Model Configuration:
# - Algorithm: RandomForestClassifier
# - Number of trees: 100
# - Max depth: 10
# - Min samples split: 5

# %%
print(f"\n{'='*60}")
print("Training Model...")
print(f"{'='*60}\n")

model = train_model(
    X_train, 
    y_train,
    n_estimators=100,
    max_depth=10,
    min_samples_split=5
)

# %% [markdown]
# ## 7. Evaluate Model Performance
# 
# Calculate comprehensive metrics:
# - Accuracy
# - Confusion Matrix
# - Precision, Recall, F1-Score

# %%
print(f"\n{'='*60}")
print("Evaluating Model...")
print(f"{'='*60}\n")

results = evaluate_model(model, X_test, y_test)
print_metrics(results)

# %% [markdown]
# ## 8. Visualize Results
# 
# Create three types of visualizations:
# 1. Actual vs Predicted Direction Timeline
# 2. Confusion Matrix Heatmap
# 3. Feature Importance Bar Chart

# %%
print(f"\n{'='*60}")
print("Creating Visualizations...")
print(f"{'='*60}\n")

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')

# 1. Actual vs Predicted Timeline
print("📊 Plot 1: Actual vs Predicted Direction")
plot_predictions(
    X_test.index,
    results['actual'].values,
    results['predictions'],
    symbol=SYMBOL,
    save_path=None,
    show=True
)

# 2. Confusion Matrix
print("\n📊 Plot 2: Confusion Matrix")
plot_confusion_matrix(
    results['confusion_matrix'],
    save_path=None,
    show=True
)

# 3. Feature Importance
print("\n📊 Plot 3: Feature Importance")
plot_feature_importance(
    model,
    X_test.columns.tolist(),
    top_n=10,
    save_path=None,
    show=True
)

# %% [markdown]
# ## 9. Feature Importance Analysis
# 
# Let's examine which features are most important for predictions:

# %%
feature_importance_df = get_feature_importance(model, X_test.columns.tolist())

print("\n📊 Feature Importance Rankings:")
print("="*60)
print(feature_importance_df.to_string(index=False))

# %% [markdown]
# ## 10. Make Predictions on New Data
# 
# Let's see how to make predictions on the most recent data:

# %%
# Get the last 10 predictions
last_10_dates = X_test.index[-10:]
last_10_actual = results['actual'].values[-10:]
last_10_predicted = results['predictions'][-10:]
last_10_proba = results['probabilities'][-10:]

print("\n🎯 Last 10 Predictions:")
print("="*60)

prediction_df = pd.DataFrame({
    'Date': last_10_dates,
    'Actual': ['Up' if x == 1 else 'Down' for x in last_10_actual],
    'Predicted': ['Up' if x == 1 else 'Down' for x in last_10_predicted],
    'Confidence': [f"{max(p):.1%}" for p in last_10_proba],
    'Correct': ['✅' if a == p else '❌' for a, p in zip(last_10_actual, last_10_predicted)]
})

print(prediction_df.to_string(index=False))

# %% [markdown]
# ## 11. Summary and Insights
# 
# ### Key Takeaways:
# 
# 1. **Model Performance**: 
#    - The model achieves reasonable accuracy for direction prediction
#    - Performance varies based on market conditions and stock volatility
# 
# 2. **Important Features**:
#    - Technical indicators like RSI, SMA, and EMA are typically most important
#    - Recent price movements (Returns, Momentum) also contribute significantly
# 
# 3. **Limitations**:
#    - Stock markets are inherently unpredictable
#    - Past performance doesn't guarantee future results
#    - This is for educational purposes only
# 
# 4. **Next Steps**:
#    - Try different stocks to see how performance varies
#    - Experiment with different model parameters
#    - Add more technical indicators
#    - Implement backtesting strategies

# %%
print("\n" + "="*60)
print("✅ Notebook Complete!")
print("="*60)
print("\n📚 To try different stocks, change the SYMBOL variable and re-run.")
print("🔧 To modify model parameters, edit the train_model() call.")
print("📊 To add more features, modify src/feature_engineering.py")
print("\n🌐 For a web interface, run: python web/app.py")
print("="*60)

# %% [markdown]
# ## Bonus: Try Multiple Stocks
# 
# Uncomment and run this cell to compare performance across different stocks:

# %%
# Compare multiple stocks
"""
stocks_to_compare = ['AAPL', 'GOOGL', 'MSFT']
results_comparison = {}

for symbol in stocks_to_compare:
    print(f"\n{'='*60}")
    print(f"Processing {symbol}...")
    print(f"{'='*60}")
    
    # Download and prepare data
    data = download_stock_data(symbol)
    if data is not None:
        X, y = prepare_features(data)
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Train and evaluate
        model = train_model(X_train, y_train)
        results = evaluate_model(model, X_test, y_test)
        
        # Store results
        results_comparison[symbol] = {
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1_score']
        }

# Display comparison
comparison_df = pd.DataFrame(results_comparison).T
print("\n📊 Performance Comparison:")
print(comparison_df)
"""
