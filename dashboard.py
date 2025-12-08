import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from models import DemandPredictor
import datetime

BRANCH_COLORS = [
    '#0EA5E9', '#10B981', '#F59E0B', '#F43F5E', 
    '#EC4899', '#06B6D4', '#F97316', '#6366F1', '#84CC16'
]

# Page configuration
st.set_page_config(
    page_title="Demand Prediction Dashboard",
    page_icon="",
    layout="wide"
)

# Initialize predictor (cache this to avoid reloading on every interaction)
@st.cache_resource
def get_predictor():
    predictor = DemandPredictor()
    return predictor

try:
    predictor = get_predictor()
except Exception as e:
    st.error(f"Error loading model/data: {e}")
    st.stop()

# Information Sidebar
st.sidebar.title("Configuration")

# Item Selection
all_items = sorted(predictor.df['item'].unique())
selected_item = st.sidebar.selectbox("Select Item", all_items)

# Prediction Days
# Prediction Days (Fixed to 7)
days_to_predict = 7

# Navigation
page = st.sidebar.radio("Navigate", ["Overview", "Predictions", "Model Comparison", "Feature Importance"])

# --- Helper Functions ---
def get_item_data(item):
    return predictor.df[predictor.df['item'] == item].sort_values('date')

# --- Pages ---

if page == "Overview":
    st.title(f"Overview: {selected_item}")
    
    item_df = get_item_data(selected_item)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Data Points", len(item_df))
    with col2:
        st.metric("Average Sales", f"{item_df['sales'].mean():.2f}")
    with col3:
        st.metric("Max Sales", item_df['sales'].max())
    with col4:
        st.metric("Date Range", f"{item_df['date'].min().date()} to {item_df['date'].max().date()}")
    
    # Time Series Plot
    fig = px.line(item_df, x='date', y='sales', title=f"Historical Sales Data for {selected_item}",
                  color_discrete_sequence=[BRANCH_COLORS[0]])
    st.plotly_chart(fig, use_container_width=True)
    
    # Seasonality Analysis (Day of Week)


elif page == "Predictions":
    st.title(f"Forecast: {selected_item}")
    
    item_df = get_item_data(selected_item)
    
    with st.spinner("Generating XGBoost predictions..."):
        predictions, mae, rmse, r2, _ = predictor.xgboost_predict(item_df, days_to_predict)
    
    # Dates for prediction
    last_date = item_df['date'].max()
    future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(days_to_predict)]
    
    # Create prediction dataframe
    pred_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Sales': predictions
    })
    
    # Metrics
    st.subheader("Model Performance (Test Set)")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{mae:.2f}")
    m2.metric("RMSE", f"{rmse:.2f}")
    m3.metric("R² Score", f"{r2:.4f}")
    
    # Plot Historical + Prediction
    # Combine for plotting
    hist_plot_df = item_df[['date', 'sales']].tail(60) # Show last 60 days of history
    hist_plot_df.columns = ['Date', 'Sales']
    hist_plot_df['Type'] = 'Historical'
    
    pred_plot_df = pred_df.copy()
    pred_plot_df.columns = ['Date', 'Sales']
    pred_plot_df['Type'] = 'Predicted'
    
    combined_df = pd.concat([hist_plot_df, pred_plot_df])
    
    fig = px.line(combined_df, x='Date', y='Sales', color='Type', 
                  title=f"Sales Forecast for {selected_item} (Next {days_to_predict} Days)",
                  color_discrete_map={"Historical": BRANCH_COLORS[0], "Predicted": BRANCH_COLORS[1]})
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Forecast Data")
    st.dataframe(pred_df)

elif page == "Model Comparison":
    st.title(f"Model Comparison: {selected_item}")
    
    item_df = get_item_data(selected_item)
    
    with st.spinner("Running all models..."):
        # Run all models
        ma_preds = predictor.moving_average_predict(item_df, days_to_predict)
        lr_preds = predictor.linear_regression_predict(item_df, days_to_predict)
        ts_preds = predictor.time_series_predict(item_df, days_to_predict)
        xgb_preds, _, _, _, _ = predictor.xgboost_predict(item_df, days_to_predict)
    
    # Prepare data for plotting
    last_date = item_df['date'].max()
    future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(days_to_predict)]
    
    comparison_df = pd.DataFrame({
        'Date': future_dates,
        'Moving Average': ma_preds,
        'Linear Regression': lr_preds,
        'Time Series (Holt-Winters)': ts_preds,
        'XGBoost': xgb_preds
    })
    
    # Melt for plotly
    melted_df = comparison_df.melt(id_vars=['Date'], var_name='Model', value_name='Predicted Sales')
    
    # Add historical data for context (last 30 days)
    hist_df = item_df.tail(30)[['date', 'sales']].copy()
    hist_df.columns = ['Date', 'Predicted Sales']
    hist_df['Model'] = 'Historical'
    
    final_plot_df = pd.concat([hist_df, melted_df])
    
    fig = px.line(final_plot_df, x='Date', y='Predicted Sales', color='Model',
                  title="Comparison of All Models",
                  color_discrete_sequence=BRANCH_COLORS)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Detailed Predictions Table")
    st.dataframe(comparison_df)

elif page == "Feature Importance":
    st.title(f"Feature Importance: {selected_item}")
    st.info("This shows which features had the most impact on the XGBoost model's predictions.")
    
    item_df = get_item_data(selected_item)
    
    with st.spinner("Calculating feature importance..."):
        _, _, _, _, top_features = predictor.xgboost_predict(item_df, 7)
    
    # Convert to df
    feat_df = pd.DataFrame(top_features, columns=['Feature', 'Importance'])
    feat_df = feat_df.head(20).sort_values('Importance', ascending=True) # Top 20
    
    fig = px.bar(feat_df, x='Importance', y='Feature', orientation='h', 
                 title="XGBoost Feature Importance",
                 color_discrete_sequence=[BRANCH_COLORS[0]])
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Feature Definitions")
    st.markdown("""
    - **sales**: Previous sales values
    - **lag_X**: Sales value X days ago
    - **rolling_mean_X**: Average sales over the last X days
    - **rolling_std_X**: Standard deviation (volatility) over the last X days
    - **daily_diff**: Difference between today's and yesterday's sales
    - **day_of_week**: The day of the week (0=Monday, 6=Sunday)
    - **is_weekend**: Whether the day is a Saturday or Sunday
    """)
