# visualize_comparison.py (Updated for XGBoost)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
import seaborn as sns

def load_and_visualize():
    """Load predictions and create comparison plots"""
    
    # Load data
    df = pd.read_csv('enhanced_sales_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    with open('all_model_predictions.json', 'r') as f:
        predictions = json.load(f)
    
    # Get last 30 days of actual data for comparison
    last_date = df['date'].max()
    start_date = last_date - timedelta(days=30)
    recent_data = df[df['date'] >= start_date]
    
    # Create visualization for each item
    items = list(predictions.keys())
    n_items = len(items)
    
    fig = plt.figure(figsize=(20, 6*n_items))
    
    # Create gridspec for more flexible layout
    gs = fig.add_gridspec(n_items, 3, hspace=0.4, wspace=0.3)
    
    for idx, item in enumerate(items):
        # Plot 1: Model comparison for next 7 days
        ax1 = fig.add_subplot(gs[idx, 0])
        
        # Get next 7 dates
        last_date = df[df['item'] == item]['date'].max()
        future_dates = [last_date + timedelta(days=i+1) for i in range(7)]
        
        # Plot predictions from each model
        model_predictions = predictions[item]
        colors = ['blue', 'green', 'red', 'orange']
        model_names = ['Moving Average', 'Linear Regression', 'Time Series', 'XGBoost']
        
        for model_name, color in zip(model_names, colors):
            if model_name in model_predictions and model_name != 'XGBoost Metrics' and model_name != 'Top Features':
                preds = model_predictions[model_name]
                ax1.plot(future_dates, preds, 'o-', label=model_name, color=color, 
                        linewidth=2.5, markersize=8, alpha=0.8)
        
        ax1.set_title(f'{item} - Model Predictions Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Predicted Demand', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 2: Historical vs Predicted
        ax2 = fig.add_subplot(gs[idx, 1])
        
        # Plot historical data (last 30 days)
        item_data = recent_data[recent_data['item'] == item]
        ax2.plot(item_data['date'], item_data['sales'], 'ko-', 
                label='Historical', linewidth=2, markersize=4)
        
        # Plot XGBoost predictions
        xgb_preds = model_predictions['XGBoost']
        ax2.plot(future_dates, xgb_preds, 'ro-', label='XGBoost Predictions', 
                linewidth=2.5, markersize=8)
        
        ax2.set_title(f'{item} - Historical vs XGBoost Predictions', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Demand', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 3: Feature Importance (XGBoost)
        ax3 = fig.add_subplot(gs[idx, 2])
        
        if 'Top Features' in model_predictions:
            top_features = model_predictions['Top Features'][:10]
            features, importance = zip(*top_features)
            
            y_pos = np.arange(len(features))
            ax3.barh(y_pos, importance, align='center', color='steelblue', alpha=0.8)
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(features, fontsize=9)
            ax3.set_xlabel('Importance', fontsize=12)
            ax3.set_title(f'{item} - XGBoost Feature Importance', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Demand Prediction Model Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('model_comparison_xgboost.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Create metrics comparison table
    print("\n" + "="*80)
    print("MODEL PERFORMANCE METRICS (XGBoost)")
    print("="*80)
    
    metrics_summary = []
    for item in items:
        model_predictions = predictions[item]
        
        if 'XGBoost Metrics' in model_predictions:
            metrics = model_predictions['XGBoost Metrics']
            
            # Calculate simple metrics for other models
            predictions_data = {}
            for model_name in ['Moving Average', 'Linear Regression', 'Time Series', 'XGBoost']:
                if model_name in model_predictions:
                    preds = model_predictions[model_name]
                    preds_array = np.array(preds)
                    
                    predictions_data[model_name] = {
                        'Mean': np.mean(preds_array),
                        'Std': np.std(preds_array),
                        'CV': (np.std(preds_array) / np.mean(preds_array)) * 100 if np.mean(preds_array) > 0 else 0
                    }
            
            metrics_summary.append({
                'Item': item,
                'XGBoost MAE': f"{metrics['MAE']:.2f}",
                'XGBoost RMSE': f"{metrics['RMSE']:.2f}",
                'XGBoost R²': f"{metrics['R2']:.3f}",
                'XGBoost Avg Prediction': f"{predictions_data.get('XGBoost', {}).get('Mean', 0):.1f}",
                'MA Avg Prediction': f"{predictions_data.get('Moving Average', {}).get('Mean', 0):.1f}",
                'LR Avg Prediction': f"{predictions_data.get('Linear Regression', {}).get('Mean', 0):.1f}",
                'TS Avg Prediction': f"{predictions_data.get('Time Series', {}).get('Mean', 0):.1f}"
            })
    
    metrics_df = pd.DataFrame(metrics_summary)
    print("\nPerformance Summary:")
    print(metrics_df.to_string(index=False))
    
    # Save detailed comparison
    detailed_data = []
    for item in items:
        model_predictions = predictions[item]
        for model_name, preds in model_predictions.items():
            if model_name not in ['XGBoost Metrics', 'Top Features']:
                for i, pred in enumerate(preds):
                    detailed_data.append({
                        'item': item,
                        'model': model_name,
                        'day': i+1,
                        'prediction': pred,
                        'date': (df[df['item'] == item]['date'].max() + timedelta(days=i+1)).strftime('%Y-%m-%d')
                    })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv('detailed_predictions_xgboost.csv', index=False)
    
    print("\n" + "="*80)
    print(f"Detailed predictions saved to 'detailed_predictions_xgboost.csv'")
    print(f"Visualization saved to 'model_comparison_xgboost.png'")
    print("="*80)
    
    return metrics_df, detailed_df

if __name__ == "__main__":
    metrics_df, detailed_df = load_and_visualize()