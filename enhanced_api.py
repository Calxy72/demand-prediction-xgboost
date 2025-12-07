# enhanced_api.py (FIXED VERSION)
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from models import DemandPredictor
import uvicorn

app = FastAPI(
    title="Advanced Demand Prediction API with XGBoost",
    description="API with 4 different prediction models including XGBoost",
    version="2.0.0"
)

# Initialize predictor
predictor = DemandPredictor()

class PredictionResponse(BaseModel):
    item: str
    model: str
    predictions: List[int]
    dates: List[str]
    metrics: Optional[Dict[str, float]] = None
    feature_importance: Optional[List[Dict[str, Any]]] = None

class ModelComparisonResponse(BaseModel):
    item: str
    comparisons: Dict[str, List[int]]
    best_model: str
    explanation: str
    xgboost_metrics: Optional[Dict[str, float]] = None

class FeatureImportanceResponse(BaseModel):
    item: str
    top_features: List[Dict[str, Any]]
    total_features: int

@app.get("/")
def read_root():
    return {
        "message": "Advanced Demand Prediction API with XGBoost",
        "status": "running",
        "available_models": ["moving_average", "linear_regression", "time_series", "xgboost"],
        "endpoints": {
            "/": "This documentation",
            "/docs": "Interactive API documentation (Swagger UI)",
            "/redoc": "Alternative API documentation",
            "/predict/{item}": "Predict using default model (XGBoost)",
            "/predict/{item}/{model}": "Predict using specific model",
            "/compare/{item}": "Compare all 4 models",
            "/features/{item}": "Get XGBoost feature importance",
            "/items": "List all items",
            "/models": "List all available models",
            "/model_recommendation": "Get recommended model for item"
        },
        "try_these_links": [
            "http://localhost:8000/items",
            "http://localhost:8000/predict/Apple",
            "http://localhost:8000/compare/Banana",
            "http://localhost:8000/docs"
        ]
    }

@app.get("/predict/{item_name}")
def predict_item_default(
    item_name: str,
    days: int = Query(7, ge=1, le=30, description="Number of days to predict")
):
    """Predict using default model (XGBoost)"""
    try:
        item_df = predictor.df[predictor.df['item'] == item_name].sort_values('date')
        
        if len(item_df) == 0:
            raise HTTPException(status_code=404, detail="Item not found")
        
        # Use XGBoost as default
        predictions, mae, rmse, r2, top_features = predictor.xgboost_predict(item_df, days)
        
        # Get dates
        last_date = item_df['date'].max()
        future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(days)]
        
        # Format feature importance (FIXED: feature is string, importance is float)
        feature_importance = [{"feature": str(f), "importance": float(i)} for f, i in top_features[:10]]
        
        return PredictionResponse(
            item=item_name,
            model="xgboost",
            predictions=predictions,
            dates=[d.strftime('%Y-%m-%d') for d in future_dates],
            metrics={
                "MAE": float(mae),
                "RMSE": float(rmse),
                "R2": float(r2)
            },
            feature_importance=feature_importance
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/{item_name}/{model_name}")
def predict_with_model(
    item_name: str,
    model_name: str,
    days: int = Query(7, ge=1, le=30, description="Number of days to predict")
):
    """Predict using specific model"""
    try:
        item_df = predictor.df[predictor.df['item'] == item_name].sort_values('date')
        
        if len(item_df) == 0:
            raise HTTPException(status_code=404, detail="Item not found")
        
        metrics = None
        feature_importance = None
        
        # Select model
        if model_name == "moving_average":
            predictions = predictor.moving_average_predict(item_df, days)
        elif model_name == "linear_regression":
            predictions = predictor.linear_regression_predict(item_df, days)
        elif model_name == "time_series":
            predictions = predictor.time_series_predict(item_df, days)
        elif model_name == "xgboost":
            predictions, mae, rmse, r2, top_features = predictor.xgboost_predict(item_df, days)
            metrics = {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}
            feature_importance = [{"feature": str(f), "importance": float(i)} for f, i in top_features[:10]]
        else:
            raise HTTPException(status_code=400, detail="Invalid model name")
        
        # Get dates
        last_date = item_df['date'].max()
        future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(days)]
        
        return PredictionResponse(
            item=item_name,
            model=model_name,
            predictions=predictions,
            dates=[d.strftime('%Y-%m-%d') for d in future_dates],
            metrics=metrics,
            feature_importance=feature_importance
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/compare/{item_name}")
def compare_models(
    item_name: str,
    days: int = Query(7, ge=1, le=30, description="Number of days to predict")
):
    """Compare predictions from all 4 models"""
    try:
        item_df = predictor.df[predictor.df['item'] == item_name].sort_values('date')
        
        if len(item_df) == 0:
            raise HTTPException(status_code=404, detail="Item not found")
        
        # Get predictions from all models
        ma_preds = predictor.moving_average_predict(item_df, days)
        lr_preds = predictor.linear_regression_predict(item_df, days)
        ts_preds = predictor.time_series_predict(item_df, days)
        xgb_preds, mae, rmse, r2, top_features = predictor.xgboost_predict(item_df, days)
        
        # Calculate model consistency
        models_data = {
            'moving_average': ma_preds,
            'linear_regression': lr_preds,
            'time_series': ts_preds,
            'xgboost': xgb_preds
        }
        
        # Calculate different metrics for model selection
        model_scores = {}
        for name, preds in models_data.items():
            preds_array = np.array(preds)
            
            # Lower std dev is better (more stable predictions)
            std_dev = np.std(preds_array)
            
            # Higher mean might indicate over/under prediction
            mean_pred = np.mean(preds_array)
            hist_mean = item_df['sales'].mean()
            mean_diff = abs(mean_pred - hist_mean)
            
            # Combined score (lower is better)
            score = std_dev + (mean_diff * 0.5)
            model_scores[name] = score
        
        # Select best model (lowest score)
        best_model = min(model_scores, key=model_scores.get)
        
        return ModelComparisonResponse(
            item=item_name,
            comparisons={
                'moving_average': ma_preds,
                'linear_regression': lr_preds,
                'time_series': ts_preds,
                'xgboost': xgb_preds
            },
            best_model=best_model,
            explanation=f"Selected {best_model} based on stability score (std dev + mean difference)",
            xgboost_metrics={
                "MAE": float(mae),
                "RMSE": float(rmse),
                "R2": float(r2)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/features/{item_name}")
def get_feature_importance(
    item_name: str,
    top_n: int = Query(10, ge=1, le=20, description="Number of top features to return")
):
    """Get XGBoost feature importance for an item"""
    try:
        item_df = predictor.df[predictor.df['item'] == item_name].sort_values('date')
        
        if len(item_df) == 0:
            raise HTTPException(status_code=404, detail="Item not found")
        
        # Train XGBoost model to get feature importance
        _, _, _, _, top_features = predictor.xgboost_predict(item_df, 7)
        
        # Limit to top_n features
        top_features = top_features[:top_n]
        
        # Format response
        formatted_features = [{"feature": str(f), "importance": float(i)} for f, i in top_features]
        
        return FeatureImportanceResponse(
            item=item_name,
            top_features=formatted_features,
            total_features=len(formatted_features)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/items")
def list_items():
    """List all available items"""
    items = predictor.df['item'].unique().tolist()
    return {"items": items, "count": len(items)}

@app.get("/models")
def list_models():
    """List all available models with descriptions"""
    models = {
        "moving_average": {
            "description": "Weighted moving average using 3, 7, and 14-day windows with EMA",
            "complexity": "Low",
            "best_for": "Stable demand patterns, quick predictions",
            "pros": "Simple, fast, interpretable",
            "cons": "Cannot capture complex patterns"
        },
        "linear_regression": {
            "description": "Linear regression with time-based features, lags, and rolling statistics",
            "complexity": "Medium",
            "best_for": "Linear trends and seasonality",
            "pros": "Interpretable coefficients, handles trends well",
            "cons": "Assumes linear relationships"
        },
        "time_series": {
            "description": "Holt-Winters exponential smoothing with weekly seasonality",
            "complexity": "Medium",
            "best_for": "Seasonal time series data",
            "pros": "Handles seasonality naturally, good for pure time series",
            "cons": "Sensitive to parameter tuning"
        },
        "xgboost": {
            "description": "XGBoost gradient boosting with advanced feature engineering",
            "complexity": "High",
            "best_for": "Complex non-linear patterns, multiple influencing factors",
            "pros": "High accuracy, handles non-linear relationships, robust to outliers",
            "cons": "Complex, slower training, harder to interpret"
        }
    }
    return models

@app.get("/model_recommendation")
def get_model_recommendation(item_name: str):
    """Get recommended model for a specific item based on data characteristics"""
    try:
        item_df = predictor.df[predictor.df['item'] == item_name].sort_values('date')
        
        if len(item_df) == 0:
            raise HTTPException(status_code=404, detail="Item not found")
        
        # Analyze data characteristics
        sales_std = item_df['sales'].std()
        sales_mean = item_df['sales'].mean()
        cv = (sales_std / sales_mean) * 100 if sales_mean > 0 else 0
        
        # Check for seasonality (day of week pattern)
        dow_variation = item_df.groupby('day_of_week')['sales'].std().mean()
        
        # Check for trend
        from scipy import stats
        x = np.arange(len(item_df))
        y = item_df['sales'].values
        slope, _, r_value, _, _ = stats.linregress(x, y)
        trend_strength = abs(r_value)
        
        # Determine recommendation
        if len(item_df) < 30:
            recommendation = "moving_average"
            reason = "Insufficient data for complex models"
        elif cv < 20:  # Low variation
            recommendation = "moving_average"
            reason = "Low variation in sales data"
        elif dow_variation > sales_std * 0.5:  # Strong day of week pattern
            recommendation = "time_series"
            reason = "Strong weekly seasonality detected"
        elif trend_strength > 0.7:  # Strong trend
            recommendation = "linear_regression"
            reason = "Strong linear trend detected"
        else:  # Complex patterns
            recommendation = "xgboost"
            reason = "Complex patterns requiring advanced modeling"
        
        return {
            "item": item_name,
            "recommended_model": recommendation,
            "reason": reason,
            "data_characteristics": {
                "data_points": len(item_df),
                "mean_sales": float(sales_mean),
                "std_sales": float(sales_std),
                "coefficient_of_variation": float(cv),
                "trend_strength": float(trend_strength),
                "dow_variation": float(dow_variation)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def print_startup_message():
    """Print helpful startup message"""
    print("\n" + "="*60)
    print("DEMAND PREDICTION API STARTED SUCCESSFULLY!")
    print("="*60)
    print("\n📊 API Endpoints:")
    print("  • http://localhost:8000/              - API Documentation")
    print("  • http://localhost:8000/docs          - Interactive Swagger UI")
    print("  • http://localhost:8000/redoc         - Alternative docs")
    print("  • http://localhost:8000/items         - List all items")
    print("  • http://localhost:8000/predict/Apple - Predict for Apple")
    print("  • http://localhost:8000/compare/Banana- Compare all models")
    print("\n🔧 Quick Tests:")
    print("  • Open browser to: http://localhost:8000")
    print("  • Or test with curl: curl http://localhost:8000/items")
    print("\n" + "="*60)
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")

if __name__ == "__main__":
    print_startup_message()
    uvicorn.run(app, host="localhost", port=8000, reload=False)