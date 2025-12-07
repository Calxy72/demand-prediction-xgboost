import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_enhanced_data(num_days=90):
    """Generate realistic sales data with multiple patterns"""
    
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    
    items = {
        'Apple': {'base': 80, 'trend': 'increasing', 'seasonality': 'low'},
        'Banana': {'base': 60, 'trend': 'stable', 'seasonality': 'medium'},
        'Tomato': {'base': 50, 'trend': 'decreasing', 'seasonality': 'high'},
        'Potato': {'base': 70, 'trend': 'stable', 'seasonality': 'low'},
        'Spinach': {'base': 40, 'trend': 'seasonal', 'seasonality': 'very_high'}
    }
    
    data = []
    for i, date in enumerate(dates):
        for item, properties in items.items():
            base_sales = properties['base']
            
            # 1. Add trend
            if properties['trend'] == 'increasing':
                trend_effect = i * 0.2  # Slowly increasing
            elif properties['trend'] == 'decreasing':
                trend_effect = -i * 0.15  # Slowly decreasing
            elif properties['trend'] == 'seasonal':
                # Seasonal pattern (monthly cycle)
                seasonal = 20 * np.sin(2 * np.pi * i / 30)
                trend_effect = seasonal
            else:
                trend_effect = 0
            
            # 2. Add seasonality effect
            if properties['seasonality'] == 'low':
                seasonal_effect = 5 * np.sin(2 * np.pi * i / 7)  # Weekly
            elif properties['seasonality'] == 'medium':
                seasonal_effect = 10 * np.sin(2 * np.pi * i / 7)  # Weekly
            elif properties['seasonality'] == 'high':
                seasonal_effect = 15 * np.sin(2 * np.pi * i / 7)  # Weekly
            elif properties['seasonality'] == 'very_high':
                seasonal_effect = 20 * np.sin(2 * np.pi * i / 7)  # Weekly
            
            # 3. Weekend effect
            weekend_effect = 0
            if date.weekday() >= 5:  # Weekend
                weekend_effect = base_sales * 0.3
            
            # 4. Random noise
            noise = np.random.normal(0, 8)
            
            # Calculate final sales
            sales = base_sales + trend_effect + seasonal_effect + weekend_effect + noise
            sales = max(10, int(sales))
            
            # Add promotional events (random)
            if random.random() < 0.05:  # 5% chance of promotion
                sales = int(sales * 1.5)
            
            data.append({
                'date': date.date(),
                'item': item,
                'sales': sales,
                'price': np.random.uniform(0.5, 3.0),
                'day_of_week': date.strftime('%A'),
                'day_of_week_num': date.weekday(),
                'is_weekend': 1 if date.weekday() >= 5 else 0,
                'month': date.month,
                'day_of_month': date.day
            })
    
    df = pd.DataFrame(data)
    df.to_csv('enhanced_sales_data.csv', index=False)
    print(f"Generated enhanced dataset with {len(df)} rows")
    return df

if __name__ == "__main__":
    generate_enhanced_data(90)  # 90 days of data