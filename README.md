# Advanced Demand Prediction System with XGBoost

## Overview
A comprehensive demand prediction system implementing 4 different modeling approaches with XGBoost as the advanced ML model.

## Key Features
1. **4 Complete Model Implementations**:
   - **Moving Averages**: Weighted averages with day-of-week adjustments
   - **Linear Regression**: Advanced feature engineering with lag variables
   - **Time Series**: Holt-Winters exponential smoothing
   - **XGBoost**: Gradient boosting with comprehensive feature engineering

2. **Advanced Feature Engineering**:
   - 30+ engineered features for XGBoost
   - Cyclical encoding for seasonality
   - Multiple lag variables (1, 2, 3, 7, 14 days)
   - Rolling statistics (mean, std, min, max)
   - Exponential moving averages
   - Difference and percent change features

3. **Model Comparison & Evaluation**:
   - Side-by-side prediction comparisons
   - XGBoost performance metrics (MAE, RMSE, R²)
   - Feature importance analysis
   - Model recommendation engine

4. **Production-Ready API**:
   - REST API with FastAPI
   - Model selection endpoint
   - Feature importance endpoint
   - Automated model recommendation

## Project Structure
```bash
demand_prediction_xgboost/
├── generate_enhanced_data.py # Generate realistic dataset
├── enhanced_sales_data.csv # Generated sales data (90 days)
├── models.py # All 4 model implementations (with XGBoost)
├── visualize_comparison.py # Model comparison visualizations
├── enhanced_api.py # REST API with XGBoost support
├── all_model_predictions.json # JSON output of all predictions
├── model_comparison_xgboost.png # Visualization image
├── detailed_predictions_xgboost.csv # Detailed predictions
├── requirements.txt # Python dependencies (with xgboost)
└── README.md # This file
```
## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Generate sample data: `python generate_enhanced_data.py`
4. Run predictions: `python models.py`
5. Start API: `python enhanced_api.py`

**Note**: The repository doesn't include generated data files. 
Run step 3 to create the dataset locally.



## Installation

```bash
# Install all dependencies (including XGBoost)
pip install -r requirements.txt

# Alternative: Install manually
pip install pandas numpy scikit-learn statsmodels fastapi uvicorn matplotlib seaborn xgboost


# 1. Generate sample data
python generate_enhanced_data.py

# 2. Run all models and compare
python models.py

# 3. Create visualizations
python visualize_comparison.py

# 4. Start API server
python enhanced_api.py
```
## XGBoost Model Details
Features Used:
Temporal Features: Day number, day of week, month, quarter, weekend flag

Cyclical Features: Sine/cosine encoding for month, day, week

Lag Features: Sales from previous 1, 2, 3, 7, 14 days

Rolling Statistics: 3, 7, 14-day means, 7-day std/min/max

EMA Features: 3 and 7-day exponential moving averages

Difference Features: 1 and 7-day differences

Target Encoding: Mean sales per day of week

## Model Performance & Limitations

### Current Performance:
- **Moving Average**: Good baseline, stable predictions
- **Linear Regression**: Captures trends well
- **Time Series**: Best for weekly patterns  
- **XGBoost**: Most sophisticated, handles complexity

### Accuracy Notes:
With the current 90-day synthetic dataset, predictions show reasonable patterns but would benefit from:
1. **More historical data** (365+ days for seasonality)
2. **Real-world data** (actual vendor sales)
3. **External factors** (weather, holidays, events)

### Scalability:
The architecture is production-ready and accuracy will improve naturally with more data collection.

# Citation

If you use this in your work, please cite:

Demand Prediction System with XGBoost. (2024). GitHub Repository.

# Contact

For questions or issues, please open an issue on the GitHub repository.