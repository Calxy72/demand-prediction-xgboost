# generate_large_dataset.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_large_dataset(num_days=730):  # 2 years = 730 days
    """Generate realistic sales data with multiple patterns"""
    
    start_date = datetime(2022, 1, 1)  # Start 2 years ago
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    
    items = {
        'Apple': {'base': 80, 'trend': 'increasing', 'seasonality': 'medium'},
        'Banana': {'base': 60, 'trend': 'stable', 'seasonality': 'high'},
        'Tomato': {'base': 50, 'trend': 'seasonal', 'seasonality': 'very_high'},
        'Potato': {'base': 70, 'trend': 'stable', 'seasonality': 'low'},
        'Spinach': {'base': 40, 'trend': 'seasonal', 'seasonality': 'extreme'},
        'Orange': {'base': 55, 'trend': 'increasing', 'seasonality': 'high'},
        'Onion': {'base': 65, 'trend': 'stable', 'seasonality': 'low'},
        'Carrot': {'base': 45, 'trend': 'decreasing', 'seasonality': 'medium'}
    }
    
    data = []
    for i, date in enumerate(dates):
        for item, properties in items.items():
            base_sales = properties['base']
            
            # 1. Long-term trend (yearly)
            if properties['trend'] == 'increasing':
                trend_effect = i * 0.1  # Slow increase over 2 years
            elif properties['trend'] == 'decreasing':
                trend_effect = -i * 0.08  # Slow decrease
            elif properties['trend'] == 'seasonal':
                # Multi-year seasonal pattern
                yearly_seasonal = 15 * np.sin(2 * np.pi * i / 365)
                trend_effect = yearly_seasonal
            else:
                trend_effect = 0
            
            # 2. Weekly seasonality (stronger)
            weekly_seasonal = 0
            if properties['seasonality'] == 'low':
                weekly_seasonal = 8 * np.sin(2 * np.pi * i / 7)
            elif properties['seasonality'] == 'medium':
                weekly_seasonal = 12 * np.sin(2 * np.pi * i / 7)
            elif properties['seasonality'] == 'high':
                weekly_seasonal = 18 * np.sin(2 * np.pi * i / 7)
            elif properties['seasonality'] == 'very_high':
                weekly_seasonal = 25 * np.sin(2 * np.pi * i / 7)
            elif properties['seasonality'] == 'extreme':
                weekly_seasonal = 35 * np.sin(2 * np.pi * i / 7)
            
            # 3. Monthly seasonality
            monthly_seasonal = 10 * np.sin(2 * np.pi * date.month / 12)
            
            # 4. Weekend effect (stronger on Saturdays)
            weekend_effect = 0
            if date.weekday() == 5:  # Saturday
                weekend_effect = base_sales * 0.5
            elif date.weekday() == 6:  # Sunday
                weekend_effect = base_sales * 0.3
            elif date.weekday() == 4:  # Friday
                weekend_effect = base_sales * 0.2
            
            # 5. Holiday effects (Christmas, Summer, etc.)
            holiday_effect = 0
            # Christmas period (Dec 15-31)
            if date.month == 12 and date.day >= 15:
                holiday_effect = base_sales * 0.4
            # Summer (June-August)
            elif date.month in [6, 7, 8]:
                if item in ['Orange', 'Apple']:  # Fruits popular in summer
                    holiday_effect = base_sales * 0.3
            
            # 6. Random noise (realistic)
            noise = np.random.normal(0, 10)
            
            # 7. Promotional events (more frequent)
            promo_effect = 1.0
            if random.random() < 0.1:  # 10% chance of promotion
                promo_effect = random.choice([1.3, 1.5, 1.8])  # 30%, 50%, or 80% increase
            
            # Calculate final sales
            sales = base_sales + trend_effect + weekly_seasonal + monthly_seasonal + weekend_effect + holiday_effect + noise
            sales = sales * promo_effect
            sales = max(5, int(sales))  # Minimum 5 units
            
            # Add price (correlated with demand)
            base_price = {
                'Apple': 2.5, 'Banana': 1.5, 'Tomato': 3.0, 
                'Potato': 1.0, 'Spinach': 2.0, 'Orange': 2.0,
                'Onion': 1.2, 'Carrot': 1.8
            }[item]
            
            # Price varies with demand
            price_variation = 0.2 * (sales / base_sales - 1)  # Price goes up with demand
            price = base_price * (1 + price_variation + np.random.normal(0, 0.1))
            
            data.append({
                'date': date.date(),
                'item': item,
                'sales': sales,
                'price': round(price, 2),
                'day_of_week': date.strftime('%A'),
                'day_of_week_num': date.weekday(),
                'is_weekend': 1 if date.weekday() >= 5 else 0,
                'month': date.month,
                'day_of_month': date.day,
                'year': date.year,
                'week_of_year': date.isocalendar().week,
                'is_holiday_season': 1 if (date.month == 12 and date.day >= 15) else 0,
                'is_summer': 1 if date.month in [6, 7, 8] else 0
            })
    
    df = pd.DataFrame(data)
    df.to_csv('large_sales_data.csv', index=False)
    print(f"📊 Generated LARGE dataset with {len(df)} rows ({num_days} days)")
    print(f"   Items: {len(items)}")
    print(f"   Date range: {dates[0].date()} to {dates[-1].date()}")
    print(f"   File saved: large_sales_data.csv")
    
    # Print summary
    print("\n📈 Item Summary:")
    for item in items.keys():
        item_data = df[df['item'] == item]
        print(f"   {item}: Avg sales = {item_data['sales'].mean():.1f}, Std = {item_data['sales'].std():.1f}")
    
    return df

if __name__ == "__main__":
    generate_large_dataset(730)  # 2 years of data