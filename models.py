# models.py (Updated with XGBoost)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class DemandPredictor:
    def __init__(self, data_path='enhanced_sales_data.csv'):
    #def __init__(self, data_path='large_sales_data.csv'):
        self.df = pd.read_csv(data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.results = {}
        self.models = {}
        
    def prepare_features(self, item_df):
        """Create features for ML models"""
        item_df = item_df.copy()
        
        # Time-based features
        item_df['day_num'] = (item_df['date'] - item_df['date'].min()).dt.days
        item_df['day_of_week'] = item_df['date'].dt.dayofweek
        item_df['month'] = item_df['date'].dt.month
        item_df['day_of_month'] = item_df['date'].dt.day
        item_df['week_of_year'] = item_df['date'].dt.isocalendar().week
        item_df['quarter'] = item_df['date'].dt.quarter
        item_df['is_weekend'] = (item_df['day_of_week'] >= 5).astype(int)
        
        # Cyclical features (for seasonality)
        item_df['month_sin'] = np.sin(2 * np.pi * item_df['month'] / 12)
        item_df['month_cos'] = np.cos(2 * np.pi * item_df['month'] / 12)
        item_df['day_sin'] = np.sin(2 * np.pi * item_df['day_of_month'] / 31)
        item_df['day_cos'] = np.cos(2 * np.pi * item_df['day_of_month'] / 31)
        item_df['week_sin'] = np.sin(2 * np.pi * item_df['week_of_year'] / 52)
        item_df['week_cos'] = np.cos(2 * np.pi * item_df['week_of_year'] / 52)
        
        # Lag features
        for lag in [1, 2, 3, 7, 14]:
            item_df[f'lag_{lag}'] = item_df['sales'].shift(lag)
        
        # Rolling statistics
        item_df['rolling_mean_3'] = item_df['sales'].rolling(3).mean()
        item_df['rolling_mean_7'] = item_df['sales'].rolling(7).mean()
        item_df['rolling_mean_14'] = item_df['sales'].rolling(14).mean()
        item_df['rolling_std_7'] = item_df['sales'].rolling(7).std()
        item_df['rolling_min_7'] = item_df['sales'].rolling(7).min()
        item_df['rolling_max_7'] = item_df['sales'].rolling(7).max()
        
        # Exponential moving averages
        item_df['ema_3'] = item_df['sales'].ewm(span=3).mean()
        item_df['ema_7'] = item_df['sales'].ewm(span=7).mean()
        
        # Difference features
        item_df['diff_1'] = item_df['sales'].diff(1)
        item_df['diff_7'] = item_df['sales'].diff(7)
        
        # Percent change
        item_df['pct_change_1'] = item_df['sales'].pct_change(1)
        item_df['pct_change_7'] = item_df['sales'].pct_change(7)
        
        # Target encoding of day of week (mean sales per day)
        day_of_week_means = item_df.groupby('day_of_week')['sales'].transform('mean')
        item_df['dow_mean_encoded'] = day_of_week_means
        
        # Drop rows with NaN from lag features
        item_df = item_df.dropna()
        
        return item_df
    
    # ============================================
    # MODEL 1: MOVING AVERAGES
    # ============================================
    def moving_average_predict(self, item_df, days_ahead=7):
        """Simple moving average prediction"""
        predictions = []
        
        # Use different window sizes
        ma_3 = item_df['sales'].rolling(3).mean().iloc[-1]
        ma_7 = item_df['sales'].rolling(7).mean().iloc[-1]
        ma_14 = item_df['sales'].rolling(14).mean().iloc[-1]
        ema_7 = item_df['sales'].ewm(span=7).mean().iloc[-1]
        
        # Weighted average (recent data more important)
        weights = [0.1, 0.3, 0.3, 0.3]  # ma_3, ma_7, ma_14, ema_7
        
        for i in range(days_ahead):
            pred = (ma_3 * weights[0] + ma_7 * weights[1] + 
                   ma_14 * weights[2] + ema_7 * weights[3])
            
            # Adjust for day of week pattern
            last_date = item_df['date'].max()
            pred_date = last_date + timedelta(days=i+1)
            
            # Day of week adjustment
            dow = pred_date.weekday()
            dow_adjustment = {
                0: 0.95,  # Monday
                1: 1.00,  # Tuesday
                2: 1.05,  # Wednesday
                3: 1.10,  # Thursday
                4: 1.15,  # Friday
                5: 1.30,  # Saturday
                6: 1.25   # Sunday
            }
            pred = pred * dow_adjustment.get(dow, 1.0)
            
            predictions.append(max(10, int(pred)))
        
        return predictions
    
    # ============================================
    # MODEL 2: LINEAR REGRESSION
    # ============================================
    def linear_regression_predict(self, item_df, days_ahead=7):
        """Linear regression with multiple features"""
        # Prepare features
        item_df = self.prepare_features(item_df)
        
        if len(item_df) < 20:
            return [item_df['sales'].mean()] * days_ahead
        
        # Features and target
        features = ['day_num', 'day_of_week', 'is_weekend', 'month',
                   'lag_1', 'lag_2', 'lag_3', 'lag_7',
                   'rolling_mean_3', 'rolling_mean_7', 'rolling_std_7',
                   'ema_7', 'diff_7', 'dow_mean_encoded']
        
        X = item_df[features].values
        y = item_df['sales'].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Store for future use
        self.models['linear_regression'] = {
            'model': model,
            'scaler': scaler,
            'features': features
        }
        
        # Prepare for prediction
        predictions = []
        last_row = item_df.iloc[-1]
        
        for i in range(days_ahead):
            # Create feature vector for future date
            future_date = last_row['date'] + timedelta(days=i+1)
            future_dow = future_date.weekday()
            
            # Calculate future features
            features_future = [
                last_row['day_num'] + i + 1,  # day_num
                future_dow,  # day_of_week
                1 if future_dow >= 5 else 0,  # is_weekend
                future_date.month,  # month
                last_row['sales'],  # lag_1
                last_row['lag_1'],  # lag_2
                last_row['lag_2'],  # lag_3
                item_df['sales'].iloc[-7] if len(item_df) >= 7 else last_row['sales'],  # lag_7
                last_row['rolling_mean_3'],
                last_row['rolling_mean_7'],
                last_row['rolling_std_7'],
                last_row['ema_7'],
                last_row['diff_7'],
                item_df[item_df['day_of_week'] == future_dow]['sales'].mean() if len(item_df[item_df['day_of_week'] == future_dow]) > 0 else last_row['dow_mean_encoded']
            ]
            
            # Scale and predict
            features_scaled = scaler.transform([features_future])
            pred = model.predict(features_scaled)[0]
            predictions.append(max(10, int(pred)))
        
        return predictions
    
    # ============================================
    # MODEL 3: TIME SERIES (HOLT-WINTERS)
    # ============================================
    def time_series_predict(self, item_df, days_ahead=7):
        """Holt-Winters exponential smoothing"""
        try:
            # Resample to daily frequency
            ts_data = item_df.set_index('date')['sales']
            
            # Handle missing dates
            all_dates = pd.date_range(start=ts_data.index.min(), 
                                      end=ts_data.index.max(), freq='D')
            ts_data = ts_data.reindex(all_dates).fillna(method='ffill')
            
            # Fit Holt-Winters model with weekly seasonality
            model = ExponentialSmoothing(
                ts_data,
                seasonal_periods=7,
                trend='add',
                seasonal='add',
                initialization_method='estimated'
            )
            
            fitted_model = model.fit(optimized=True)
            
            # Make predictions
            forecast = fitted_model.forecast(days_ahead)
            predictions = [max(10, int(x)) for x in forecast]
            
            return predictions
            
        except Exception as e:
            print(f"Time series model failed: {e}")
            # Fallback to moving average
            return self.moving_average_predict(item_df, days_ahead)
    
    # ============================================
    # MODEL 4: MACHINE LEARNING (XGBoost)
    # ============================================
    def xgboost_predict(self, item_df, days_ahead=7):
        """XGBoost Regressor with advanced feature engineering"""
        # Prepare features
        item_df = self.prepare_features(item_df)
        
        if len(item_df) < 40:
            print(f"Insufficient data for XGBoost, using Linear Regression")
            return self.linear_regression_predict(item_df, days_ahead)
        
        # Define feature columns
        feature_cols = [
            'day_num', 'day_of_week', 'month', 'quarter', 'is_weekend',
            'month_sin', 'month_cos', 'day_sin', 'day_cos', 'week_sin', 'week_cos',
            'lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14',
            'rolling_mean_3', 'rolling_mean_7', 'rolling_mean_14',
            'rolling_std_7', 'rolling_min_7', 'rolling_max_7',
            'ema_3', 'ema_7',
            'diff_1', 'diff_7',
            'pct_change_1', 'pct_change_7',
            'dow_mean_encoded'
        ]
        
        X = item_df[feature_cols].values
        y = item_df['sales'].values
        
        # Train-test split for evaluation (last 20% for test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # XGBoost parameters
        params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse'
        }
        
        # Train XGBoost model
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
        )
        
        # Evaluate on test set
        y_pred_test = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        r2 = r2_score(y_test, y_pred_test)
        
        # Feature importance
        feature_importance = dict(zip(feature_cols, model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Store model
        self.models['xgboost'] = {
            'model': model,
            'feature_cols': feature_cols,
            'importance': top_features
        }
        
        # Prepare future predictions
        predictions = []
        last_row = item_df.iloc[-1]
        
        for i in range(days_ahead):
            # Create feature vector for future date
            future_date = last_row['date'] + timedelta(days=i+1)
            
            # Calculate all features for future date
            future_features = self.calculate_future_features(
                last_row, item_df, future_date, feature_cols
            )
            
            # Predict
            pred = model.predict([future_features])[0]
            predictions.append(max(10, int(pred)))
        
        return predictions, mae, rmse, r2, top_features
    
    def calculate_future_features(self, last_row, item_df, future_date, feature_cols):
        """Calculate feature values for a future date"""
        features = {}
        
        # Basic time features
        features['day_num'] = last_row['day_num'] + (future_date - last_row['date']).days
        features['day_of_week'] = future_date.weekday()
        features['month'] = future_date.month
        features['quarter'] = (future_date.month - 1) // 3 + 1
        features['is_weekend'] = 1 if future_date.weekday() >= 5 else 0
        
        # Cyclical features
        features['month_sin'] = np.sin(2 * np.pi * future_date.month / 12)
        features['month_cos'] = np.cos(2 * np.pi * future_date.month / 12)
        features['day_sin'] = np.sin(2 * np.pi * future_date.day / 31)
        features['day_cos'] = np.cos(2 * np.pi * future_date.day / 31)
        features['week_sin'] = np.sin(2 * np.pi * future_date.isocalendar().week / 52)
        features['week_cos'] = np.cos(2 * np.pi * future_date.isocalendar().week / 52)
        
        # Lag features (using last known values - simplified)
        features['lag_1'] = last_row['sales']
        features['lag_2'] = last_row['lag_1']
        features['lag_3'] = last_row['lag_2']
        features['lag_7'] = item_df['sales'].iloc[-7] if len(item_df) >= 7 else last_row['sales']
        features['lag_14'] = item_df['sales'].iloc[-14] if len(item_df) >= 14 else last_row['sales']
        
        # Rolling statistics (using last known)
        features['rolling_mean_3'] = last_row['rolling_mean_3']
        features['rolling_mean_7'] = last_row['rolling_mean_7']
        features['rolling_mean_14'] = last_row['rolling_mean_14']
        features['rolling_std_7'] = last_row['rolling_std_7']
        features['rolling_min_7'] = last_row['rolling_min_7']
        features['rolling_max_7'] = last_row['rolling_max_7']
        
        # EMA features
        features['ema_3'] = last_row['ema_3']
        features['ema_7'] = last_row['ema_7']
        
        # Difference features
        features['diff_1'] = last_row['diff_1']
        features['diff_7'] = last_row['diff_7']
        
        # Percent change
        features['pct_change_1'] = last_row['pct_change_1']
        features['pct_change_7'] = last_row['pct_change_7']
        
        # Target encoding
        dow = future_date.weekday()
        dow_sales = item_df[item_df['day_of_week'] == dow]['sales']
        features['dow_mean_encoded'] = dow_sales.mean() if len(dow_sales) > 0 else last_row['dow_mean_encoded']
        
        # Convert to array in correct order
        return [features[col] for col in feature_cols]
    
    # ============================================
    # UNIFIED PREDICTION METHOD
    # ============================================
    def predict_all_models(self, days_ahead=7):
        """Run all 4 models and compare results"""
        
        all_predictions = {}
        model_names = ['Moving Average', 'Linear Regression', 'Time Series', 'XGBoost']
        
        for item in self.df['item'].unique():
            print(f"\n{'='*50}")
            print(f"Predicting for: {item}")
            print(f"{'='*50}")
            
            # Get item data
            item_df = self.df[self.df['item'] == item].sort_values('date')
            
            if len(item_df) < 14:
                print(f"  Skipping {item}: insufficient data")
                continue
            
            # Run all models
            print("Running models...")
            ma_preds = self.moving_average_predict(item_df, days_ahead)
            lr_preds = self.linear_regression_predict(item_df, days_ahead)
            ts_preds = self.time_series_predict(item_df, days_ahead)
            xgb_preds, mae, rmse, r2, top_features = self.xgboost_predict(item_df, days_ahead)
            
            # Store results
            all_predictions[item] = {
                'Moving Average': ma_preds,
                'Linear Regression': lr_preds,
                'Time Series': ts_preds,
                'XGBoost': xgb_preds,
                'XGBoost Metrics': {
                    'MAE': float(mae),
                    'RMSE': float(rmse),
                    'R2': float(r2)
                },
                'Top Features': top_features
            }
            
            # Print results
            print(f"\nPredictions for next {days_ahead} days:")
            print(f"  Moving Average:    {ma_preds}")
            print(f"  Linear Regression: {lr_preds}")
            print(f"  Time Series:       {ts_preds}")
            print(f"  XGBoost:           {xgb_preds}")
            
            print(f"\nXGBoost Performance:")
            print(f"  MAE:  {mae:.2f}")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  R²:   {r2:.3f}")
            
            print(f"\nTop 10 Important Features:")
            for feature, importance in top_features[:10]:
                print(f"  {feature}: {importance:.4f}")
        
        self.results = all_predictions
        return all_predictions
    
    def get_next_dates(self, days_ahead=7):
        """Get the next N dates for predictions"""
        last_date = self.df['date'].max()
        return [last_date + timedelta(days=i+1) for i in range(days_ahead)]

if __name__ == "__main__":
    # Test the models
    predictor = DemandPredictor()
    predictions = predictor.predict_all_models(7)
    
    # Save results
    import json
    with open('all_model_predictions.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_data = {}
        for item, models in predictions.items():
            json_data[item] = {}
            for model_name, preds in models.items():
                if model_name == 'XGBoost Metrics':
                    json_data[item][model_name] = preds
                elif model_name == 'Top Features':
                    json_data[item][model_name] = [(f, float(i)) for f, i in preds]
                else:
                    json_data[item][model_name] = [int(p) for p in preds]
        
        json.dump(json_data, f, indent=2)
    
    print("\n" + "="*60)
    print("All predictions saved to 'all_model_predictions.json'")
    print("="*60)