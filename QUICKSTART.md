# Stock Movement Direction Predictor - Quick Start Guide

## Installation & Setup

Follow these steps to get started with the Stock Movement Direction Predictor:

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Web Application

```bash
cd web
python app.py
```

Then open your browser to: **http://localhost:5000**

### 3. Use the Jupyter Notebook

```bash
jupyter notebook
```

Navigate to `notebooks/main.ipynb` and run the cells.

## Quick Test

To quickly test if everything is working:

```python
# Test data loading
from src.data_loader import download_stock_data
data = download_stock_data('AAPL')
print(f"Downloaded {len(data)} rows of data")

# Test feature engineering
from src.feature_engineering import prepare_features
X, y = prepare_features(data)
print(f"Features shape: {X.shape}")

# Test model training
from src.model import train_model, split_data
X_train, X_test, y_train, y_test = split_data(X, y)
model = train_model(X_train, y_train)
print("Model trained successfully!")
```

## Available Stock Symbols

- AAPL (Apple Inc.)
- GOOGL (Alphabet Inc.)
- MSFT (Microsoft Corporation)
- AMZN (Amazon.com Inc.)
- TSLA (Tesla Inc.)
- NVDA (NVIDIA Corporation)
- META (Meta Platforms Inc.)
- NFLX (Netflix Inc.)
- AMD (Advanced Micro Devices Inc.)
- JPM (JPMorgan Chase & Co.)
- V (Visa Inc.)

## Project Structure Overview

```
src/
├── data_loader.py          # Download stock data
├── feature_engineering.py  # Create technical indicators
├── model.py               # Train ML model
├── evaluate.py            # Evaluate performance
└── visualize.py           # Create plots

web/
├── app.py                 # Flask backend
├── templates/index.html   # Dashboard UI
└── static/
    ├── css/style.css      # Styling
    └── js/main.js         # Frontend logic
```

## Next Steps

1. **Explore the Web Dashboard**: Select stocks and view predictions
2. **Read the Notebook**: Learn how the model works step-by-step
3. **Experiment**: Try different stocks and parameters
4. **Customize**: Add your own features or indicators

For detailed documentation, see **README.md**
