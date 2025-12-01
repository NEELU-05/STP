/**
 * Stock Movement Direction Predictor
 * Frontend JavaScript - Handles UI interactions and API calls
 */

// Global variables
let charts = {
    predictions: null,
    confusion: null,
    importance: null
};

// API Base URL
const API_BASE = window.location.origin;

/**
 * Initialize the application
 */
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Stock Predictor App Initialized');
    loadStocks();
    setupEventListeners();
});

/**
 * Load available stock symbols
 */
async function loadStocks() {
    try {
        const response = await fetch(`${API_BASE}/api/stocks`);
        const data = await response.json();

        if (data.success) {
            const select = document.getElementById('stockSelect');
            select.innerHTML = '<option value="">Select a stock...</option>';

            data.data.forEach(symbol => {
                const option = document.createElement('option');
                option.value = symbol;
                option.textContent = symbol;
                select.appendChild(option);
            });

            document.getElementById('predictBtn').disabled = false;
            console.log('✅ Loaded stock symbols:', data.data);
        }
    } catch (error) {
        console.error('❌ Error loading stocks:', error);
        showError('Failed to load stock symbols. Please refresh the page.');
    }
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    // Predict button click
    document.getElementById('predictBtn').addEventListener('click', handlePredict);

    // Stock selection change
    document.getElementById('stockSelect').addEventListener('change', (e) => {
        const btn = document.getElementById('predictBtn');
        btn.disabled = !e.target.value;
    });

    // Tab switching
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            switchTab(e.target.dataset.tab);
        });
    });
}

/**
 * Handle predict button click
 */
async function handlePredict() {
    const symbol = document.getElementById('stockSelect').value;

    if (!symbol) {
        showError('Please select a stock symbol');
        return;
    }

    // Show loading state
    setLoadingState(true);

    try {
        console.log(`📊 Requesting prediction for ${symbol}...`);

        const response = await fetch(`${API_BASE}/api/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ symbol })
        });

        const result = await response.json();

        if (result.success) {
            console.log('✅ Prediction received:', result.data);
            displayResults(result.data);

            if (result.cached) {
                console.log('ℹ️ Using cached results');
            }
        } else {
            showError(result.error || 'Prediction failed');
        }
    } catch (error) {
        console.error('❌ Error during prediction:', error);
        showError('An error occurred. Please try again.');
    } finally {
        setLoadingState(false);
    }
}

/**
 * Display prediction results
 */
function displayResults(data) {
    // Show results section
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Update accuracy
    updateAccuracy(data.accuracy);

    // Update direction
    updateDirection(data.direction, data.confidence);

    // Update metrics
    updateMetrics(data.metrics);

    // Update info
    updateInfo(data);

    // Render charts
    renderCharts(data);
}

/**
 * Update accuracy display with circular progress
 */
function updateAccuracy(accuracy) {
    const percentage = Math.round(accuracy * 100);
    const circle = document.getElementById('progressCircle');
    const valueElement = document.getElementById('accuracyValue');

    // Calculate stroke-dashoffset (326.73 is the circumference)
    const circumference = 326.73;
    const offset = circumference - (percentage / 100) * circumference;

    // Animate the circle
    setTimeout(() => {
        circle.style.strokeDashoffset = offset;
    }, 100);

    // Animate the number
    animateValue(valueElement, 0, percentage, 1000, '%');
}

/**
 * Update direction badge
 */
function updateDirection(direction, confidence) {
    const badge = document.getElementById('directionBadge');
    const icon = document.getElementById('directionIcon');
    const text = document.getElementById('directionText');
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceValue = document.getElementById('confidenceValue');

    // Update badge
    badge.className = 'direction-badge ' + direction.toLowerCase();
    icon.textContent = direction === 'Up' ? '↑' : '↓';
    text.textContent = direction;

    // Update confidence
    const confidencePercent = Math.round(confidence * 100);
    setTimeout(() => {
        confidenceBar.style.width = confidencePercent + '%';
    }, 100);

    animateValue(confidenceValue, 0, confidencePercent, 800, '%');
}

/**
 * Update metrics cards
 */
function updateMetrics(metrics) {
    animateValue(
        document.getElementById('precisionValue'),
        0,
        Math.round(metrics.precision * 100),
        800,
        '%'
    );

    animateValue(
        document.getElementById('recallValue'),
        0,
        Math.round(metrics.recall * 100),
        800,
        '%'
    );

    animateValue(
        document.getElementById('f1Value'),
        0,
        Math.round(metrics.f1_score * 100),
        800,
        '%'
    );
}

/**
 * Update dataset info
 */
function updateInfo(data) {
    document.getElementById('infoSymbol').textContent = data.symbol;
    document.getElementById('infoSamples').textContent = data.total_samples.toLocaleString();
    document.getElementById('infoTestSamples').textContent = data.test_samples.toLocaleString();
    document.getElementById('infoDateRange').textContent =
        `${data.date_range.start} to ${data.date_range.end}`;
}

/**
 * Render all charts
 */
function renderCharts(data) {
    renderPredictionsChart(data.chart_data);
    renderConfusionMatrix(data.confusion_matrix);
    renderFeatureImportance(data.feature_importance);
}

/**
 * Render predictions timeline chart
 */
function renderPredictionsChart(chartData) {
    const ctx = document.getElementById('predictionsChart').getContext('2d');

    // Destroy existing chart
    if (charts.predictions) {
        charts.predictions.destroy();
    }

    charts.predictions = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartData.dates,
            datasets: [
                {
                    label: 'Actual Direction',
                    data: chartData.actual,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderWidth: 2,
                    stepped: true,
                    tension: 0
                },
                {
                    label: 'Predicted Direction',
                    data: chartData.predicted,
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    stepped: true,
                    tension: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    labels: {
                        color: '#f1f5f9',
                        font: { size: 12 }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            return context.dataset.label + ': ' +
                                (context.parsed.y === 1 ? 'Up' : 'Down');
                        }
                    }
                }
            },
            scales: {
                x: {
                    ticks: {
                        color: '#94a3b8',
                        maxRotation: 45,
                        minRotation: 45
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    }
                },
                y: {
                    ticks: {
                        color: '#94a3b8',
                        stepSize: 1,
                        callback: function (value) {
                            return value === 1 ? 'Up' : 'Down';
                        }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    },
                    min: -0.1,
                    max: 1.1
                }
            }
        }
    });
}

/**
 * Render confusion matrix
 */
/**
 * Render confusion matrix as Heatmap
 */
function renderConfusionMatrix(matrix) {
    const container = document.querySelector('.confusion-matrix-container');
    container.innerHTML = ''; // Clear existing content

    // Create Heatmap Grid
    const grid = document.createElement('div');
    grid.className = 'heatmap-grid';

    // Extract values
    // matrix = [[TN, FP], [FN, TP]]
    const tn = matrix[0][0];
    const fp = matrix[0][1];
    const fn = matrix[1][0];
    const tp = matrix[1][1];

    const total = tn + fp + fn + tp;

    const getPercent = (val) => ((val / total) * 100).toFixed(1) + '%';

    // HTML Structure
    grid.innerHTML = `
        <!-- Header Row -->
        <div class="heatmap-header"></div>
        <div class="heatmap-header">Pred: Down</div>
        <div class="heatmap-header">Pred: Up</div>
        
        <!-- Row 1: Actual Down -->
        <div class="heatmap-header">Actual: Down</div>
        <div class="heatmap-cell correct">
            <span class="heatmap-value">${tn}</span>
            <span class="heatmap-label">True Down (${getPercent(tn)})</span>
        </div>
        <div class="heatmap-cell incorrect">
            <span class="heatmap-value">${fp}</span>
            <span class="heatmap-label">False Up (${getPercent(fp)})</span>
        </div>
        
        <!-- Row 2: Actual Up -->
        <div class="heatmap-header">Actual: Up</div>
        <div class="heatmap-cell incorrect">
            <span class="heatmap-value">${fn}</span>
            <span class="heatmap-label">False Down (${getPercent(fn)})</span>
        </div>
        <div class="heatmap-cell correct">
            <span class="heatmap-value">${tp}</span>
            <span class="heatmap-label">True Up (${getPercent(tp)})</span>
        </div>
    `;

    container.appendChild(grid);

    // Remove the charts.confusion reference since we are not using Chart.js anymore
    charts.confusion = null;
}

/**
 * Render feature importance chart
 */
function renderFeatureImportance(featureData) {
    const ctx = document.getElementById('importanceChart').getContext('2d');

    // Destroy existing chart
    if (charts.importance) {
        charts.importance.destroy();
    }

    charts.importance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: featureData.features,
            datasets: [{
                label: 'Importance Score',
                data: featureData.importance,
                backgroundColor: '#8b5cf6',
                borderColor: '#8b5cf6',
                borderWidth: 2
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            return `Importance: ${context.parsed.x.toFixed(4)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    beginAtZero: true
                },
                y: {
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }
                }
            }
        }
    });
}

/**
 * Switch between tabs
 */
function switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');

    // Update tab panes
    document.querySelectorAll('.tab-pane').forEach(pane => {
        pane.classList.remove('active');
    });

    const tabMap = {
        'predictions': 'predictionsTab',
        'confusion': 'confusionTab',
        'importance': 'importanceTab'
    };

    document.getElementById(tabMap[tabName]).classList.add('active');
}

/**
 * Animate number value
 */
function animateValue(element, start, end, duration, suffix = '') {
    const range = end - start;
    const increment = range / (duration / 16); // 60fps
    let current = start;

    const timer = setInterval(() => {
        current += increment;
        if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
            current = end;
            clearInterval(timer);
        }
        element.textContent = Math.round(current) + suffix;
    }, 16);
}

/**
 * Set loading state
 */
function setLoadingState(isLoading) {
    const btn = document.getElementById('predictBtn');
    const btnText = btn.querySelector('.btn-text');
    const spinner = btn.querySelector('.spinner');
    const select = document.getElementById('stockSelect');

    if (isLoading) {
        btn.disabled = true;
        select.disabled = true;
        btnText.textContent = 'Processing...';
        spinner.style.display = 'inline-block';
    } else {
        btn.disabled = false;
        select.disabled = false;
        btnText.textContent = 'Predict Direction';
        spinner.style.display = 'none';
    }
}

/**
 * Show error message
 */
function showError(message) {
    alert('❌ Error: ' + message);
    console.error('Error:', message);
}

// Add SVG gradient for circular progress (inject into DOM)
window.addEventListener('load', () => {
    const svg = document.querySelector('.progress-ring');
    if (svg) {
        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        const gradient = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
        gradient.setAttribute('id', 'gradient');
        gradient.setAttribute('x1', '0%');
        gradient.setAttribute('y1', '0%');
        gradient.setAttribute('x2', '100%');
        gradient.setAttribute('y2', '100%');

        const stop1 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
        stop1.setAttribute('offset', '0%');
        stop1.setAttribute('style', 'stop-color:#667eea;stop-opacity:1');

        const stop2 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
        stop2.setAttribute('offset', '100%');
        stop2.setAttribute('style', 'stop-color:#764ba2;stop-opacity:1');

        gradient.appendChild(stop1);
        gradient.appendChild(stop2);
        defs.appendChild(gradient);
        svg.insertBefore(defs, svg.firstChild);
    }
});
