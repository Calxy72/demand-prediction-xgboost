# visualize_comparison.py (Fixed - No emojis for Windows compatibility)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

def create_output_folder():
    """Create output folder if it doesn't exist"""
    output_dir = 'outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[INFO] Created outputs folder: {output_dir}/")
    
    # Create subfolders for organization
    subfolders = ['charts', 'tables', 'predictions', 'comparisons']
    for folder in subfolders:
        path = os.path.join(output_dir, folder)
        if not os.path.exists(path):
            os.makedirs(path)
    
    return output_dir

def load_and_visualize():
    """Load predictions and create comparison plots"""
    
    # Create output folder
    output_dir = create_output_folder()
    
    # Load data
    try:
        df = pd.read_csv('enhanced_sales_data.csv')
        print("[INFO] Loaded enhanced_sales_data.csv")
    except FileNotFoundError:
        try:
            df = pd.read_csv('large_sales_data.csv')
            print("[INFO] Loaded large_sales_data.csv")
        except FileNotFoundError:
            print("[ERROR] No data file found. Please generate data first.")
            return None, None
    
    df['date'] = pd.to_datetime(df['date'])
    
    try:
        with open('all_model_predictions.json', 'r') as f:
            predictions = json.load(f)
        print("[INFO] Loaded predictions from all_model_predictions.json")
    except FileNotFoundError:
        print("[ERROR] all_model_predictions.json not found. Run models.py first!")
        return None, None
    
    # Get last 30 days of actual data for comparison
    last_date = df['date'].max()
    start_date = last_date - timedelta(days=30)
    recent_data = df[df['date'] >= start_date]
    
    # Create visualization for each item
    items = list(predictions.keys())
    n_items = len(items)
    
    if n_items == 0:
        print("[ERROR] No items found in predictions")
        return None, None
    
    # ============================================
    # MAIN COMPARISON PLOT (All items in one figure)
    # ============================================
    print("[CHART] Creating main comparison plot...")
    fig = plt.figure(figsize=(20, 6*n_items))
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
        model_predictions = predictions[item]
        
        # ============================================
        # PLOT 1: Model predictions comparison (future only)
        # ============================================
        ax1 = fig.add_subplot(gs[idx, 0])
        
        colors = {'Moving Average': 'blue', 
                 'Linear Regression': 'green', 
                 'Time Series': 'red', 
                 'XGBoost': 'orange'}
        
        for model_name, color in colors.items():
            if model_name in model_predictions and model_name != 'XGBoost Metrics' and model_name != 'Top Features':
                preds = model_predictions[model_name]
                if isinstance(preds, list) and len(preds) == 7:
                    ax1.plot(range(1, 8), preds, 'o-', color=color, 
                            linewidth=2.5, markersize=8, alpha=0.8, label=model_name)
        
        ax1.set_title(f'{item} - Model Predictions', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Days Ahead', fontsize=12)
        ax1.set_ylabel('Predicted Demand', fontsize=12)
        ax1.set_xticks(range(1, 8))
        ax1.set_xticklabels([f'Day {i}' for i in range(1, 8)])
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # ============================================
        # PLOT 2: ALL Models vs Historical Data
        # ============================================
        ax2 = fig.add_subplot(gs[idx, 1])
        
        # Plot historical data (last 30 days)
        item_recent_data = recent_data[recent_data['item'] == item]
        if len(item_recent_data) > 0:
            ax2.plot(item_recent_data['date'], item_recent_data['sales'], 'ko-', 
                    label='Historical (30 days)', linewidth=2, markersize=4, alpha=0.7)
        
        # Plot predictions from ALL models
        for model_name, color in colors.items():
            if model_name in model_predictions and model_name != 'XGBoost Metrics' and model_name != 'Top Features':
                preds = model_predictions[model_name]
                if isinstance(preds, list) and len(preds) == 7:
                    ax2.plot(future_dates, preds, 'o-', color=color, 
                            linewidth=2, markersize=6, alpha=0.8, label=f'{model_name}')
        
        ax2.set_title(f'{item} - All Models vs Historical', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Demand', fontsize=12)
        ax2.legend(fontsize=9, loc='upper left')
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # ============================================
        # PLOT 3: Feature Importance (XGBoost)
        # ============================================
        ax3 = fig.add_subplot(gs[idx, 2])
        
        if 'Top Features' in model_predictions and model_predictions['Top Features']:
            top_features = model_predictions['Top Features']
            
            # Handle different formats
            if isinstance(top_features[0], list):
                features = [str(f[0]) for f in top_features[:8]]
                importance = [float(f[1]) for f in top_features[:8]]
            elif isinstance(top_features[0], dict):
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
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, importance)):
                width = bar.get_width()
                ax3.text(width, bar.get_y() + bar.get_height()/2, 
                        f'{val:.3f}', ha='left', va='center', fontsize=8)
    
    plt.suptitle('Demand Prediction - All Models Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save main comparison plot
    main_plot_path = os.path.join(output_dir, 'charts', 'all_models_comparison.png')
    plt.savefig(main_plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[OK] Saved main comparison to: {main_plot_path}")
    
    # ============================================
    # INDIVIDUAL MODEL PLOTS (Separate for each model)
    # ============================================
    print("\n[CHART] Creating individual model comparison plots...")
    
    for item in items:
        item_data = df[df['item'] == item].sort_values('date')
        if len(item_data) == 0:
            continue
            
        last_date = item_data['date'].max()
        if pd.isna(last_date):
            continue
            
        future_dates = [last_date + timedelta(days=i+1) for i in range(7)]
        model_predictions = predictions[item]
        
        # Create individual plot for this item
        fig_ind, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        model_names = ['Moving Average', 'Linear Regression', 'Time Series', 'XGBoost']
        colors = ['blue', 'green', 'red', 'orange']
        
        for i, (model_name, color) in enumerate(zip(model_names, colors)):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Plot historical data
            item_recent_data = recent_data[recent_data['item'] == item]
            if len(item_recent_data) > 0:
                ax.plot(item_recent_data['date'], item_recent_data['sales'], 'ko-', 
                       label='Historical (30 days)', linewidth=1.5, markersize=3, alpha=0.7)
            
            # Plot model predictions
            if model_name in model_predictions and model_name != 'XGBoost Metrics' and model_name != 'Top Features':
                preds = model_predictions[model_name]
                if isinstance(preds, list) and len(preds) == 7:
                    ax.plot(future_dates, preds, 'o-', color=color, 
                           linewidth=2, markersize=6, alpha=0.8, label=f'{model_name}')
                    
                    # Add prediction values
                    for j, (date, pred) in enumerate(zip(future_dates, preds)):
                        ax.text(date, pred, f'{pred}', fontsize=9, 
                               ha='center', va='bottom', color=color)
            
            ax.set_title(f'{item} - {model_name}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Demand')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.suptitle(f'{item} - Individual Model Predictions', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save individual plot
        individual_path = os.path.join(output_dir, 'charts', f'{item}_individual_models.png')
        plt.savefig(individual_path, dpi=150, bbox_inches='tight')
        plt.close(fig_ind)
        print(f"[OK] Saved {item} individual plot to: {individual_path}")
    
    # ============================================
    # MODEL PERFORMANCE SUMMARY TABLE
    # ============================================
    print("\n" + "="*80)
    print("MODEL PERFORMANCE METRICS")
    print("="*80)
    
    metrics_summary = []
    for item in items:
        model_predictions = predictions[item]
        
        item_summary = {'Item': item}
        
        # XGBoost metrics
        if 'XGBoost Metrics' in model_predictions:
            metrics = model_predictions['XGBoost Metrics']
            item_summary.update({
                'XGBoost_MAE': f"{metrics.get('MAE', 0):.2f}",
                'XGBoost_RMSE': f"{metrics.get('RMSE', 0):.2f}",
                'XGBoost_R2': f"{metrics.get('R2', metrics.get('R²', 0)):.3f}"
            })
        
        # All model predictions
        for model_name in ['Moving Average', 'Linear Regression', 'Time Series', 'XGBoost']:
            if model_name in model_predictions and isinstance(model_predictions[model_name], list):
                preds = model_predictions[model_name]
                if preds:
                    avg_pred = np.mean(preds)
                    std_pred = np.std(preds)
                    item_summary[f'{model_name.replace(" ", "_")}_Avg'] = f"{avg_pred:.1f}"
                    item_summary[f'{model_name.replace(" ", "_")}_Std'] = f"{std_pred:.1f}"
        
        metrics_summary.append(item_summary)
    
    if metrics_summary:
        metrics_df = pd.DataFrame(metrics_summary)
        
        # Save metrics to CSV
        metrics_path = os.path.join(output_dir, 'tables', 'model_performance_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        
        print("\n[TABLE] Performance Summary:")
        print(metrics_df.to_string(index=False))
        print(f"\n[OK] Metrics saved to: {metrics_path}")
    else:
        print("[WARNING] No performance metrics found")
        metrics_df = pd.DataFrame()
    
    # ============================================
    # DETAILED PREDICTIONS (All models, all days)
    # ============================================
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
        
        # Save detailed predictions
        detailed_path = os.path.join(output_dir, 'predictions', 'detailed_predictions_all_models.csv')
        detailed_df.to_csv(detailed_path, index=False)
        
        print(f"[OK] Detailed predictions saved to: {detailed_path}")
    else:
        print("[WARNING] No detailed predictions generated")
        detailed_df = pd.DataFrame()
    
    # ============================================
    # CREATE MODEL COMPARISON TABLE (Console + File)
    # ============================================
    comparison_data = []
    for item in items:
        model_data = predictions[item]
        
        for model_name in ['Moving Average', 'Linear Regression', 'Time Series', 'XGBoost']:
            if model_name in model_data and isinstance(model_data[model_name], list):
                preds = model_data[model_name][:7]
                if len(preds) == 7:
                    for i, pred in enumerate(preds):
                        comparison_data.append({
                            'item': item,
                            'model': model_name,
                            'day': i+1,
                            'prediction': pred
                        })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        
        # Pivot for better view
        pivot_df = comparison_df.pivot_table(
            index=['item', 'model'], 
            columns='day', 
            values='prediction',
            aggfunc='first'
        )
        
        # Save comparison table
        comparison_path = os.path.join(output_dir, 'comparisons', 'model_predictions_table.csv')
        pivot_df.to_csv(comparison_path)
        
        print(f"[OK] Model comparison table saved to: {comparison_path}")
    
    # ============================================
    # CREATE PREDICTION SUMMARY BY DAY
    # ============================================
    if detailed_data:
        summary_by_day = detailed_df.groupby(['item', 'day']).agg({
            'prediction': ['mean', 'std', 'min', 'max']
        }).round(1)
        
        summary_path = os.path.join(output_dir, 'tables', 'daily_prediction_summary.csv')
        summary_by_day.to_csv(summary_path)
        
        print(f"[OK] Daily prediction summary saved to: {summary_path}")
    
    # ============================================
    # CREATE README FOR OUTPUTS FOLDER
    # ============================================
    readme_content = f"""# Output Files Summary

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Folder Structure:
outputs/
├── charts/                    # Visualization plots
│   ├── all_models_comparison.png      # Main comparison chart
│   └── [item]_individual_models.png   # Individual model charts
├── tables/                    # Performance metrics
│   ├── model_performance_metrics.csv   # Model performance metrics
│   └── daily_prediction_summary.csv    # Daily predictions summary
├── predictions/               # Detailed predictions
│   └── detailed_predictions_all_models.csv
└── comparisons/              # Comparison tables
    └── model_predictions_table.csv

## Data Summary:
- Items analyzed: {len(items)}
- Models compared: 4 (Moving Average, Linear Regression, Time Series, XGBoost)
- Prediction horizon: 7 days
- Historical data: Last 30 days used for comparison

## Files Description:
1. charts/all_models_comparison.png: Main comparison plot showing all models
2. charts/[item]_individual_models.png: Individual plots for each item
3. tables/model_performance_metrics.csv: Performance metrics for all models
4. tables/daily_prediction_summary.csv: Summary of daily predictions
5. predictions/detailed_predictions_all_models.csv: Detailed predictions
6. comparisons/model_predictions_table.csv: Comparison table in CSV format
"""

    readme_path = os.path.join(output_dir, 'README.md')
    try:
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print(f"[OK] Output summary saved to: {readme_path}")
    except Exception as e:
        print(f"[WARNING] Could not save README: {e}")
        # Try without special encoding
        with open(readme_path, 'w') as f:
            f.write("# Output Files Summary\n\nGenerated on: " + 
                   datetime.now().strftime('%Y-%m-%d %H:%M:%S') + 
                   "\n\nSee other files for results.")
    
    return metrics_df, detailed_df

def create_model_comparison_table():
    """Create a nice comparison table of all models (console output)"""
    try:
        with open('all_model_predictions.json', 'r') as f:
            predictions = json.load(f)
    except FileNotFoundError:
        print("[ERROR] No predictions file found")
        return
    
    print("\n" + "="*90)
    print("MODEL PREDICTIONS COMPARISON TABLE")
    print("="*90)
    
    for item in predictions.keys():
        print(f"\n{item}:")
        print("-" * 40)
        
        model_data = predictions[item]
        
        # Create a table header
        print(f"{'Model':<20} {'Day 1':<8} {'Day 2':<8} {'Day 3':<8} {'Day 4':<8} {'Day 5':<8} {'Day 6':<8} {'Day 7':<8} {'Avg':<8}")
        print("-" * 90)
        
        for model_name in ['Moving Average', 'Linear Regression', 'Time Series', 'XGBoost']:
            if model_name in model_data and isinstance(model_data[model_name], list):
                preds = model_data[model_name][:7]
                if len(preds) == 7:
                    avg_pred = sum(preds) / len(preds)
                    pred_str = '  '.join([f'{p:6.1f}' for p in preds])
                    print(f"{model_name:<20} {pred_str} {avg_pred:8.1f}")
        
        # Show XGBoost metrics
        if 'XGBoost Metrics' in model_data:
            metrics = model_data['XGBoost Metrics']
            print("\nXGBoost Performance:")
            print(f"  MAE:  {metrics.get('MAE', 0):.2f}")
            print(f"  RMSE: {metrics.get('RMSE', 0):.2f}")
            print(f"  R2:   {metrics.get('R2', metrics.get('R²', 0)):.3f}")

if __name__ == "__main__":
    print("="*60)
    print("Starting visualization generation...")
    print("="*60)
    
    # Create output folder
    create_output_folder()
    
    # Run main visualization
    metrics_df, detailed_df = load_and_visualize()
    
    # Create comparison table in console
    create_model_comparison_table()
    
    print("\n" + "="*60)
    print("SUCCESS: Visualization generation complete!")
    print("All outputs saved in 'outputs/' folder")
    print("="*60)
    
    # Show folder structure
    print("\nOutput Folder Structure:")
    print("outputs/")
    for root, dirs, files in os.walk('outputs'):
        level = root.replace('outputs', '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")