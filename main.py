import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import yfinance as yf
from datetime import datetime, timedelta

import streamlit as st

projection_days = 14

st.title(f'BTC Price Projection {projection_days}-Day')
st.header("Using LSTM+CNN and XGBoost")

# Load BTC data
btc_data = yf.download("BTC-USD", start="2025-01-01", end=datetime.now().strftime("%Y-%m-%d"))

btc_data = btc_data.reset_index()  # Make Date a regular column
btc_data.columns = btc_data.columns.droplevel(1)  # Remove the Ticker level from column
btc_data = btc_data[['Date', 'Close']]

btc_data['Date'] = btc_data['Date'].apply(lambda x : pd.to_datetime(x))

btc_future_data = btc_data.copy()

# Load LSTM data
lstm_data = pd.read_csv("lstm_cnn_btc_price.csv")

lstm_data['Date'] = lstm_data['Date'].apply(lambda x : pd.to_datetime(x))

# Load XGBoost data
xgboost_data = pd.read_csv("xgboost_btc_price.csv")

xgboost_data['Date'] = xgboost_data['Date'].apply(lambda x : pd.to_datetime(x))

# Calculate SMA with future
window = 5

# Project next X days by repeating the last SMA value (naive forecast)
future_dates = [pd.to_datetime(pd.Timestamp(btc_data['Date'].iloc[-1]).timestamp() + 86400 * i, unit='s') for i in range(projection_days)]

for i in range(projection_days):
    next_date = future_dates[i]
    
    btc_future_data['SMA'] = btc_future_data['Close'].rolling(window=window).mean()
    last_sma = btc_future_data['SMA'].iloc[-1]

    new_row = pd.DataFrame({'Date': [next_date], 'Close': [last_sma], 'SMA': [last_sma]})
    btc_future_data = pd.concat([btc_future_data, new_row], ignore_index=True)

# Plot
import plotly.graph_objects as go
import plotly.express as px

today = pd.to_datetime(datetime.now().date())
#st.write(f"Today's Date: {today.strftime('%Y-%m-%d')}")

# Replace your matplotlib code with this:

fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(
    x=btc_data['Date'][-50:],
    y=btc_data['Close'][-50:],
    name='Real Price',
    line=dict(color='white'),
    hovertemplate='Date: %{x|%Y-%m-%d}<br>Price: $%{y:.2f}<extra></extra>'
))

fig.add_trace(go.Scatter(
    x=lstm_data['Date'][:projection_days],
    y=lstm_data['Predicted_Price'][:projection_days],
    name='LSTM+CNN',
    line=dict(color='red'),
    hovertemplate='Date: %{x|%Y-%m-%d}<br>Price: $%{y:.2f}<extra></extra>'
))

fig.add_trace(go.Scatter(
    x=xgboost_data['Date'][:projection_days],
    y=xgboost_data['Predicted_Price'][:projection_days],
    name='XGBoost',
    line=dict(color='blue'),
    hovertemplate='Date: %{x|%Y-%m-%d}<br>Price: $%{y:.2f}<extra></extra>'
))

fig.add_trace(go.Scatter(
    x=btc_future_data['Date'][-50-projection_days:],
    y=btc_future_data['SMA'][-50-projection_days:],
    name='SMA',
    line=dict(color='green', dash='dash'),
    hovertemplate='Date: %{x|%Y-%m-%d}<br>Price: $%{y:.2f}<extra></extra>'
))

# Add vertical line for today's date
fig.add_shape(
    type="line",
    x0=today, y0=0,
    x1=today, y1=1,
    yref="paper",  # Makes line span entire y-axis
    line=dict(
        color="gray",
        width=2,
        dash="dash"
    )
)

# Update layout
fig.update_layout(
    title='BTC Price',
    xaxis_title='Date',
    yaxis_title='Price (USD)',
    hovermode='x unified'  # Shows all values at the same x position
)

# Display in Streamlit
st.plotly_chart(fig, use_container_width=True)