"""
Flask Web Application for Stock Movement Direction Predictor
Provides RESTful API and web dashboard for stock predictions.
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import sys
import os
import json

# Add parent directory to path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import download_stock_data, get_available_symbols
from src.feature_engineering import prepare_features
from src.model import train_model, split_data, get_feature_importance
from src.evaluate import evaluate_model
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for API calls

# Cache for trained models (in-memory storage)
model_cache = {}


@app.route('/')
def index():
    """Serve the main dashboard page."""
    return render_template('index.html')


@app.route('/api/stocks', methods=['GET'])
def get_stocks():
    """
    Get list of available stock symbols.
    
    Returns
    -------
    JSON response with stock symbols
    """
    try:
        symbols = get_available_symbols()
        return jsonify({
            'success': True,
            'data': symbols
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/predict', methods=['POST'])
def predict_stock():
    """
    Train model and make predictions for a stock symbol.
    
    Request Body
    ------------
    {
        "symbol": "AAPL"
    }
    
    Returns
    -------
    JSON response with prediction results and metrics
    """
    try:
        # Get symbol from request
        data = request.get_json()
        symbol = data.get('symbol', '').upper()
        
        if not symbol:
            return jsonify({
                'success': False,
                'error': 'Stock symbol is required'
            }), 400
        
        print(f"\n{'='*60}")
        print(f"🎯 Processing prediction request for: {symbol}")
        print(f"{'='*60}")
        
        # Check if model is already cached
        if symbol in model_cache:
            print(f"✅ Using cached model for {symbol}")
            cached_data = model_cache[symbol]
            return jsonify({
                'success': True,
                'data': cached_data,
                'cached': True
            })
        
        # Step 1: Download stock data
        stock_data = download_stock_data(symbol)
        
        if stock_data is None or stock_data.empty:
            return jsonify({
                'success': False,
                'error': f'Failed to download data for {symbol}. Please check if the symbol is valid.'
            }), 404
        
        # Step 2: Prepare features
        X, y = prepare_features(stock_data)
        
        if len(X) < 100:
            return jsonify({
                'success': False,
                'error': f'Insufficient data for {symbol}. Need at least 100 samples after feature engineering.'
            }), 400
        
        # Step 3: Split data
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        
        # Step 4: Train model
        model = train_model(X_train, y_train)
        
        # Step 5: Evaluate model
        results = evaluate_model(model, X_test, y_test)
        
        # Step 6: Get feature importance
        feature_importance = get_feature_importance(model, X_test.columns.tolist())
        
        # Step 7: Prepare response data
        # Get latest prediction (most recent test sample)
        latest_prediction = int(results['predictions'][-1])
        latest_confidence = float(np.max(results['probabilities'][-1]))
        
        # Prepare chart data for actual vs predicted
        chart_data = {
            'dates': [str(date.date()) for date in X_test.index[-50:]],  # Last 50 days
            'actual': results['actual'].values[-50:].tolist(),
            'predicted': results['predictions'][-50:].tolist()
        }
        
        # Prepare feature importance data (top 10)
        feature_imp_data = {
            'features': feature_importance.head(10)['feature'].tolist(),
            'importance': feature_importance.head(10)['importance'].tolist()
        }
        
        response_data = {
            'symbol': symbol,
            'accuracy': float(results['accuracy']),
            'direction': 'Up' if latest_prediction == 1 else 'Down',
            'confidence': latest_confidence,
            'metrics': {
                'precision': float(results['precision']),
                'recall': float(results['recall']),
                'f1_score': float(results['f1_score'])
            },
            'confusion_matrix': results['confusion_matrix'].tolist(),
            'chart_data': chart_data,
            'feature_importance': feature_imp_data,
            'total_samples': len(stock_data),
            'test_samples': len(X_test),
            'date_range': {
                'start': str(stock_data.index[0].date()),
                'end': str(stock_data.index[-1].date())
            }
        }
        
        # Cache the results
        model_cache[symbol] = response_data
        
        print(f"\n✅ Prediction complete for {symbol}")
        print(f"   Accuracy: {results['accuracy']:.2%}")
        print(f"   Latest Prediction: {response_data['direction']}")
        print(f"{'='*60}\n")
        
        return jsonify({
            'success': True,
            'data': response_data,
            'cached': False
        })
        
    except Exception as e:
        print(f"\n❌ Error processing prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/clear-cache', methods=['POST'])
def clear_cache():
    """Clear the model cache."""
    try:
        model_cache.clear()
        return jsonify({
            'success': True,
            'message': 'Cache cleared successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'cached_models': len(model_cache)
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Starting Stock Movement Direction Predictor Web App")
    print("="*60)
    print("\n📊 Dashboard URL: http://localhost:5000")
    print("🔌 API Endpoints:")
    print("   GET  /api/stocks       - Get available stock symbols")
    print("   POST /api/predict      - Predict stock direction")
    print("   POST /api/clear-cache  - Clear model cache")
    print("   GET  /health           - Health check")
    print("\n" + "="*60 + "\n")
    
    # Use environment variable to determine if in production
    import os
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    app.run(debug=debug, host='0.0.0.0', port=port)
