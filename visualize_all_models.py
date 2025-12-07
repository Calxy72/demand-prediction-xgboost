# visualize_comparison.py (Updated - All Models vs Historical)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_and_visualize():
    """Load predictions and create comparison plots"""
    
    # Load data
    try:
        df = pd.read_csv('enhanced_sales_data.csv')
        print("📊 Loaded enhanced_sales_data.csv")
    except FileNotFoundError:
        try:
            df = pd.read_csv('large_sales_data.csv')
            print("📊 Loaded large_sales_data.csv")
        except FileNotFoundError:
            print("❌ No data file found. Please generate data first.")
            return None, None
    
    df['date'] = pd.to_datetime(df['date'])
    
    try:
        with open('all_model_predictions.json', 'r') as f:
            predictions = json.load(f)
        print("✅ Loaded predictions from all_model_predictions.json")
    except FileNotFoundError:
        print("❌ all_model_predictions.json not found. Run models.py first!")
        return None, None
    
    # Get last 30 days of actual data for comparison
    last_date = df['date'].max()
    start_date = last_date - timedelta(days=30)
    recent_data = df[df['date'] >= start_date]
    
    # Create visualization for each item
    items = list(predictions.keys())
    n_items = len(items)
    
    if n_items == 0:
        print("❌ No items found in predictions")
        return None, None
    
    # Create figure with 3 columns: 
    # 1. Model predictions comparison
    # 2. ALL models vs historical
    # 3. Feature importance
    fig = plt.figure(figsize=(20, 6*n_items))
    
    # Create gridspec for more flexible layout
    gs = fig.add_gridspec(n_items, 3, hspace=0.4, wspace=0.3)
    
    for idx, item in enumerate(items):
        # Get item data
        item_data = df[df['item'] == item].sort_values('date')
        if len(item_data) == 0:
            continue
            
        last_date = item_data['date'].max()
        if pd.isna(last_date):
            continue
            
        future_dates = [last_date + timedelta(days=i+1) for i in range(7)]
        future_dates_str = [d.strftime('%Y-%m-%d') for d in future_dates]
        
        model_predictions = predictions[item]
        
        # ============================================
        # PLOT 1: Model predictions comparison (future only)
        # ============================================
        ax1 = fig.add_subplot(gs[idx, 0])
        
        colors = {'Moving Average': 'blue', 
                 'Linear Regression': 'green', 
                 'Time Series': 'red', 
                 'XGBoost': 'orange'}
        
        legend_labels = []
        for model_name, color in colors.items():
            if model_name in model_predictions and model_name != 'XGBoost Metrics' and model_name != 'Top Features':
                preds = model_predictions[model_name]
                if isinstance(preds, list) and len(preds) == 7:
                    ax1.plot(range(1, 8), preds, 'o-', color=color, 
                            linewidth=2.5, markersize=8, alpha=0.8, label=model_name)
                    legend_labels.append(model_name)
        
        ax1.set_title(f'{item} - Model Predictions Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Days Ahead', fontsize=12)
        ax1.set_ylabel('Predicted Demand', fontsize=12)
        ax1.set_xticks(range(1, 8))
        ax1.set_xticklabels([f'Day {i}' for i in range(1, 8)])
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # ============================================
        # PLOT 2: ALL Models vs Historical Data
        # ============================================
        ax2 = fig.add_subplot(gs[idx, 1])
        
        # Plot historical data (last 30 days)
        item_recent_data = recent_data[recent_data['item'] == item]
        if len(item_recent_data) > 0:
            ax2.plot(item_recent_data['date'], item_recent_data['sales'], 'ko-', 
                    label='Historical (Last 30 days)', linewidth=2, markersize=4, alpha=0.7)
        
        # Plot predictions from ALL models
        # We'll offset each model slightly for clarity
        date_offset = {
            'Moving Average': timedelta(hours=-6),
            'Linear Regression': timedelta(hours=-3),
            'Time Series': timedelta(hours=0),
            'XGBoost': timedelta(hours=3)
        }
        
        for model_name, color in colors.items():
            if model_name in model_predictions and model_name != 'XGBoost Metrics' and model_name != 'Top Features':
                preds = model_predictions[model_name]
                if isinstance(preds, list) and len(preds) == 7:
                    # Apply slight time offset for each model
                    offset_dates = [d + date_offset[model_name] for d in future_dates]
                    
                    # Plot line
                    ax2.plot(offset_dates, preds, 'o-', color=color, 
                            linewidth=2, markersize=6, alpha=0.8, label=f'{model_name} Predictions')
                    
                    # Add text labels for first and last predictions
                    if len(preds) > 0:
                        # First prediction
                        ax2.text(offset_dates[0], preds[0], f'{preds[0]}', 
                                fontsize=9, ha='center', va='bottom', color=color)
                        # Last prediction
                        ax2.text(offset_dates[-1], preds[-1], f'{preds[-1]}', 
                                fontsize=9, ha='center', va='bottom', color=color)
        
        ax2.set_title(f'{item} - Historical vs All Model Predictions', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Demand', fontsize=12)
        ax2.legend(fontsize=9, loc='upper left')
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # ============================================
        # PLOT 3: Feature Importance (XGBoost) or Model Performance
        # ============================================
        ax3 = fig.add_subplot(gs[idx, 2])
        
        # Option A: Show XGBoost feature importance
        if 'Top Features' in model_predictions and model_predictions['Top Features']:
            top_features = model_predictions['Top Features']
            
            # Handle different formats
            if isinstance(top_features[0], list):  # List of [feature, importance]
                features = [str(f[0]) for f in top_features[:8]]
                importance = [float(f[1]) for f in top_features[:8]]
            elif isinstance(top_features[0], dict):  # List of dicts
                features = [str(f.get('feature', f'Feature_{i}')) for i, f in enumerate(top_features[:8])]
                importance = [float(f.get('importance', 0)) for f in top_features[:8]]
            else:
                features = [str(f) for f in range(min(8, len(top_features)))]
                importance = [float(f) for f in top_features[:8]]
            
            y_pos = np.arange(len(features))
            bars = ax3.barh(y_pos, importance, align='center', color='steelblue', alpha=0.8)
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(features, fontsize=9)
            ax3.set_xlabel('Importance', fontsize=12)
            ax3.set_title(f'{item} - XGBoost Feature Importance', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='x')
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, importance)):
                width = bar.get_width()
                ax3.text(width, bar.get_y() + bar.get_height()/2, 
                        f'{val:.3f}', ha='left', va='center', fontsize=8)
        
        # Option B: If no feature importance, show model performance comparison
        elif 'XGBoost Metrics' in model_predictions:
            # Create a simple bar chart comparing model predictions
            models = []
            avg_predictions = []
            
            for model_name in colors.keys():
                if model_name in model_predictions and isinstance(model_predictions[model_name], list):
                    preds = model_predictions[model_name]
                    if preds:
                        avg_pred = np.mean(preds)
                        models.append(model_name)
                        avg_predictions.append(avg_pred)
            
            if models:
                bars = ax3.bar(models, avg_predictions, 
                              color=[colors[m] for m in models], alpha=0.8)
                ax3.set_title(f'{item} - Average Predictions', fontsize=14, fontweight='bold')
                ax3.set_ylabel('Average Demand')
                ax3.grid(True, alpha=0.3)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}', ha='center', va='bottom')
                
                # Rotate x labels if needed
                plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Demand Prediction Model Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('model_comparison_all_models.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ============================================
    # CREATE SEPARATE PLOT: Each model vs historical individually
    # ============================================
    print("\n📊 Creating individual model comparison plots...")
    
    for item in items:
        item_data = df[df['item'] == item].sort_values('date')
        if len(item_data) == 0:
            continue
            
        last_date = item_data['date'].max()
        if pd.isna(last_date):
            continue
            
        future_dates = [last_date + timedelta(days=i+1) for i in range(7)]
        model_predictions = predictions[item]
        
        # Create a 2x2 grid for each model
        fig2, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        model_names = ['Moving Average', 'Linear Regression', 'Time Series', 'XGBoost']
        colors = ['blue', 'green', 'red', 'orange']
        
        for i, (model_name, color) in enumerate(zip(model_names, colors)):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Plot historical data (last 30 days)
            item_recent_data = recent_data[recent_data['item'] == item]
            if len(item_recent_data) > 0:
                ax.plot(item_recent_data['date'], item_recent_data['sales'], 'ko-', 
                       label='Historical (Last 30 days)', linewidth=1.5, markersize=3, alpha=0.7)
            
            # Plot model predictions
            if model_name in model_predictions and model_name != 'XGBoost Metrics' and model_name != 'Top Features':
                preds = model_predictions[model_name]
                if isinstance(preds, list) and len(preds) == 7:
                    ax.plot(future_dates, preds, 'o-', color=color, 
                           linewidth=2, markersize=6, alpha=0.8, label=f'{model_name} Predictions')
                    
                    # Add prediction values as text
                    for j, (date, pred) in enumerate(zip(future_dates, preds)):
                        ax.text(date, pred, f'{pred}', fontsize=9, 
                               ha='center', va='bottom', color=color)
            
            ax.set_title(f'{item} - {model_name} Predictions', fontsize=12, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Demand')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.suptitle(f'{item} - Individual Model Comparisons', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{item}_individual_models.png', dpi=150, bbox_inches='tight')
        plt.close(fig2)  # Close the figure to free memory
        print(f"  ✅ Saved {item}_individual_models.png")
    
    # ============================================
    # Performance metrics table
    # ============================================
    print("\n" + "="*80)
    print("MODEL PERFORMANCE METRICS")
    print("="*80)
    
    metrics_summary = []
    for item in items:
        model_predictions = predictions[item]
        
        item_summary = {'Item': item}
        
        # XGBoost metrics if available
        if 'XGBoost Metrics' in model_predictions:
            metrics = model_predictions['XGBoost Metrics']
            item_summary.update({
                'XGBoost MAE': f"{metrics.get('MAE', 0):.2f}",
                'XGBoost RMSE': f"{metrics.get('RMSE', 0):.2f}",
                'XGBoost R²': f"{metrics.get('R2', metrics.get('R²', 0)):.3f}"
            })
        
        # Average predictions for all models
        for model_name in ['Moving Average', 'Linear Regression', 'Time Series', 'XGBoost']:
            if model_name in model_predictions and isinstance(model_predictions[model_name], list):
                preds = model_predictions[model_name]
                if preds:
                    avg_pred = np.mean(preds)
                    std_pred = np.std(preds)
                    item_summary[f'{model_name} Avg'] = f"{avg_pred:.1f}"
                    item_summary[f'{model_name} Std'] = f"{std_pred:.1f}"
        
        metrics_summary.append(item_summary)
    
    if metrics_summary:
        metrics_df = pd.DataFrame(metrics_summary)
        print("\n📈 Performance Summary:")
        print(metrics_df.to_string(index=False))
    else:
        print("⚠️ No performance metrics found")
        metrics_df = pd.DataFrame()
    
    # Save detailed predictions
    detailed_data = []
    for item in items:
        model_predictions = predictions[item]
        
        for model_name, preds in model_predictions.items():
            if model_name not in ['XGBoost Metrics', 'Top Features'] and isinstance(preds, list):
                for i, pred in enumerate(preds):
                    if i < 7:
                        last_date = df[df['item'] == item]['date'].max()
                        if not pd.isna(last_date):
                            pred_date = last_date + timedelta(days=i+1)
                            detailed_data.append({
                                'item': item,
                                'model': model_name,
                                'day': i+1,
                                'prediction': pred,
                                'date': pred_date.strftime('%Y-%m-%d')
                            })
    
    if detailed_data:
        detailed_df = pd.DataFrame(detailed_data)
        detailed_df.to_csv('detailed_predictions_all_models.csv', index=False)
        
        print("\n" + "="*80)
        print(f"✅ Detailed predictions saved to 'detailed_predictions_all_models.csv'")
        print(f"✅ Main visualization saved to 'model_comparison_all_models.png'")
        print(f"✅ Individual model plots saved as '{{item}}_individual_models.png'")
        print("="*80)
    else:
        print("⚠️ No detailed predictions generated")
        detailed_df = pd.DataFrame()
    
    return metrics_df, detailed_df

def create_model_comparison_table():
    """Create a nice comparison table of all models"""
    try:
        with open('all_model_predictions.json', 'r') as f:
            predictions = json.load(f)
    except FileNotFoundError:
        print("❌ No predictions file found")
        return
    
    print("\n" + "="*90)
    print("MODEL PREDICTIONS COMPARISON TABLE")
    print("="*90)
    
    for item in predictions.keys():
        print(f"\n📊 {item}:")
        print("-" * 40)
        
        model_data = predictions[item]
        
        # Create a table header
        print(f"{'Model':<20} {'Day 1':<8} {'Day 2':<8} {'Day 3':<8} {'Day 4':<8} {'Day 5':<8} {'Day 6':<8} {'Day 7':<8} {'Avg':<8}")
        print("-" * 90)
        
        for model_name in ['Moving Average', 'Linear Regression', 'Time Series', 'XGBoost']:
            if model_name in model_data and isinstance(model_data[model_name], list):
                preds = model_data[model_name][:7]  # Only first 7 days
                if len(preds) == 7:
                    avg_pred = sum(preds) / len(preds)
                    pred_str = '  '.join([f'{p:6.1f}' for p in preds])
                    print(f"{model_name:<20} {pred_str} {avg_pred:8.1f}")
        
        # Show XGBoost metrics if available
        if 'XGBoost Metrics' in model_data:
            metrics = model_data['XGBoost Metrics']
            print("\n📈 XGBoost Performance:")
            print(f"  MAE:  {metrics.get('MAE', 0):.2f}")
            print(f"  RMSE: {metrics.get('RMSE', 0):.2f}")
            print(f"  R²:   {metrics.get('R2', metrics.get('R²', 0)):.3f}")

if __name__ == "__main__":
    print("📊 Starting visualization generation...")
    print("="*60)
    
    # Run main visualization
    metrics_df, detailed_df = load_and_visualize()
    
    # Create comparison table
    create_model_comparison_table()
    
    print("\n" + "="*60)
    print("✅ Visualization generation complete!")
    print("="*60)