# Stock Movement Direction Predictor

A production-ready machine learning project that predicts stock price direction (Up/Down) using technical indicators and Random Forest Classifier. Features both a Jupyter notebook for learning and a modern web dashboard for predictions.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.2+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 Features

- **10+ Pre-configured Stocks**: AAPL, GOOGL, MSFT, AMZN, TSLA, NVDA, META, NFLX, AMD, JPM, V
- **All Historical Data**: Fetches maximum available data from Yahoo Finance
- **Technical Indicators**: SMA, EMA, RSI, Momentum, Returns
- **Random Forest Classifier**: Robust ML model with 100 estimators
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Interactive Visualizations**: Predictions timeline, confusion matrix, feature importance
- **Modern Web Dashboard**: Dark theme with glassmorphism effects
- **RESTful API**: Flask backend with CORS support
- **Beginner-Friendly**: Jupyter notebook with detailed explanations

## 📁 Project Structure

```
stock_direction_predictor/
├── data/                    # Auto-created for downloaded stock data
├── src/                     # Core Python modules
│   ├── __init__.py
│   ├── data_loader.py       # Download stock data from Yahoo Finance
│   ├── feature_engineering.py  # Create technical indicators
│   ├── model.py             # Train RandomForestClassifier
│   ├── evaluate.py          # Evaluate model performance
│   └── visualize.py         # Create visualizations
├── web/                     # Flask web application
│   ├── app.py              # Flask server
│   ├── static/
│   │   ├── css/style.css   # Custom styling
│   │   └── js/main.js      # Frontend logic
│   └── templates/
│       └── index.html      # Dashboard UI
├── notebooks/
│   └── main.ipynb          # Complete workflow demonstration
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Installation

1. **Clone or download the project**

```bash
cd stock_direction_predictor
```

2. **Create a virtual environment (recommended)**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

### Running the Web Application

1. **Start the Flask server**

```bash
cd web
python app.py
```

2. **Open your browser**

Navigate to: `http://localhost:5000`

## 🚀 Deployment

This application is ready to deploy on Render or any other cloud platform.

### Quick Deploy to Render

1. Push your code to GitHub
2. Create a new Web Service on [Render](https://render.com)
3. Connect your repository
4. Use these settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `cd web && gunicorn app:app`
   - **Environment**: Python 3

For detailed deployment instructions, see **[DEPLOYMENT.md](DEPLOYMENT.md)**

### Environment Variables

- `PORT`: Server port (default: 5000)
- `FLASK_ENV`: Set to `production` for production mode

Navigate to: `http://localhost:5000`


3. **Use the dashboard**

- Select a stock symbol from the dropdown
- Click "Predict Direction"
- View results, metrics, and visualizations

### Running the Jupyter Notebook

1. **Start Jupyter**

```bash
jupyter notebook
```

2. **Open the notebook**

Navigate to `notebooks/main.ipynb`

3. **Run cells**

Execute cells sequentially to see the complete workflow

## 📊 How It Works

### 1. Data Loading

- Downloads historical OHLCV data from Yahoo Finance
- Fetches all available historical data (typically 10+ years)
- Handles missing values and validates data quality

### 2. Feature Engineering

Creates technical indicators:
- **SMA (10, 20, 50)**: Simple Moving Averages
- **EMA (12, 26)**: Exponential Moving Averages
- **RSI (14)**: Relative Strength Index
- **Momentum (10)**: Price rate of change
- **Returns**: Daily percentage returns

### 3. Target Variable

Binary classification:
- **1 (Up)**: Next day's close > current close
- **0 (Down)**: Next day's close ≤ current close

### 4. Model Training

- **Algorithm**: RandomForestClassifier
- **Parameters**: 100 estimators, max_depth=10
- **Split**: 80% training, 20% testing (time-series aware)
- **No shuffle**: Maintains temporal order

### 5. Evaluation

Comprehensive metrics:
- Accuracy score
- Confusion matrix
- Precision, Recall, F1-Score
- Feature importance analysis

### 6. Visualization

- Actual vs Predicted timeline
- Confusion matrix heatmap
- Feature importance bar chart

## 🎨 Web Dashboard Features

### Modern UI Design

- **Dark Theme**: Easy on the eyes with #0f172a background
- **Glassmorphism**: Semi-transparent cards with backdrop blur
- **Gradient Accents**: Purple/blue gradients for visual appeal
- **Smooth Animations**: Fade-ins, slide-ups, loading spinners
- **Responsive**: Works on mobile, tablet, and desktop

### Interactive Elements

- Stock symbol dropdown with 10+ options
- Real-time prediction with loading states
- Circular progress indicator for accuracy
- Direction badge (Up ↑ / Down ↓) with color coding
- Confidence percentage meter
- Tabbed visualizations using Chart.js

### API Endpoints

- `GET /api/stocks` - Get available stock symbols
- `POST /api/predict` - Predict stock direction
- `POST /api/clear-cache` - Clear model cache
- `GET /health` - Health check

## 📈 Example Usage

### Python Script

```python
from src.data_loader import download_stock_data
from src.feature_engineering import prepare_features
from src.model import train_model, split_data
from src.evaluate import evaluate_model, print_metrics

# Download data
data = download_stock_data('AAPL')

# Prepare features
X, y = prepare_features(data)

# Split data
X_train, X_test, y_train, y_test = split_data(X, y)

# Train model
model = train_model(X_train, y_train)

# Evaluate
results = evaluate_model(model, X_test, y_test)
print_metrics(results)
```

### API Request

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL"}'
```

## 🔧 Configuration

### Modify Stock Symbols

Edit `src/data_loader.py`:

```python
DEFAULT_SYMBOLS = [
    'AAPL', 'GOOGL', 'MSFT',
    # Add your symbols here
]
```

### Adjust Model Parameters

Edit `src/model.py`:

```python
model = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Maximum tree depth
    min_samples_split=5,   # Minimum samples to split
    random_state=42
)
```

### Change Data Period

By default, all historical data is fetched. To limit:

```python
data = download_stock_data(
    'AAPL',
    start_date='2020-01-01',
    end_date='2023-12-31'
)
```

## 🐛 Troubleshooting

### Issue: "No data found for symbol"

- **Solution**: Verify the stock symbol is valid on Yahoo Finance
- Try a different symbol from the default list

### Issue: "Insufficient data"

- **Solution**: The stock may be newly listed or delisted
- Need at least 100 samples after feature engineering

### Issue: "Module not found"

- **Solution**: Ensure all dependencies are installed
```bash
pip install -r requirements.txt
```

### Issue: "Port 5000 already in use"

- **Solution**: Change the port in `web/app.py`
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

## 📚 Technical Details

### Dependencies

- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **yfinance**: Stock data download
- **scikit-learn**: Machine learning
- **matplotlib/seaborn**: Visualizations
- **flask**: Web framework
- **Chart.js**: Interactive charts

### Model Performance

Typical accuracy ranges from 55-75% depending on:
- Stock volatility
- Market conditions
- Amount of historical data
- Feature quality

**Note**: This is for educational purposes. Stock prediction is inherently uncertain. Do not use for actual trading decisions.

## 🎓 Learning Path

1. **Start with the notebook**: `notebooks/main.ipynb`
2. **Understand each module**: Read docstrings in `src/`
3. **Experiment with parameters**: Try different stocks and settings
4. **Explore the web app**: See how Flask integrates with ML
5. **Customize**: Add new features or indicators

## 🚧 Future Improvements

- [ ] Add more technical indicators (MACD, Bollinger Bands)
- [ ] Implement ensemble methods (XGBoost, LightGBM)
- [ ] Add backtesting framework
- [ ] Support for cryptocurrency data
- [ ] Real-time predictions with live data
- [ ] User authentication and saved predictions
- [ ] Export predictions to CSV/PDF
- [x] Modern UI Design with Glassmorphism

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Yahoo Finance** for providing free stock data via yfinance
- **scikit-learn** for excellent ML library
- **Flask** for lightweight web framework
- **Chart.js** for beautiful interactive charts

## 📞 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the code documentation

---

**⚠️ Disclaimer**: This project is for educational purposes only. Stock market predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always consult with a financial advisor before making investment choices.

**Built with ❤️ using Python, Flask, and Machine Learning**
