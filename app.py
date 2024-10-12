import streamlit as st
from datetime import date, timedelta
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
import pickle
import numpy as np
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(page_title="Bitcoin Price Prediction", layout="wide")

# Set start date for data and today's date
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
TODAY_DATE = pd.to_datetime(TODAY)

# Set Bitcoin symbol for predictions
selected_stock = 'BTC-USD'  # Bitcoin

# Fixed prediction period of 5 days
period = 5

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

@st.cache_resource
def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def generate_future_features(last_known_data, future_dates):
    return pd.DataFrame({
        'day': [d.day for d in future_dates],
        'month': [d.month for d in future_dates],
        'year': [d.year for d in future_dates],
        'SMA_10': last_known_data['SMA_10'],
        'SMA_30': last_known_data['SMA_30'],
        'EMA_10': last_known_data['EMA_10'],
        'EMA_30': last_known_data['EMA_30']
    })

# Streamlit app
st.title('Bitcoin Price Prediction')

# Load data
data = load_data(selected_stock)

# Load the pre-trained model
model_filename = 'BTC_rf_model_with_moving_avg.pkl'
best_rf = load_model(model_filename)

# Prepare the data for predictions
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Feature Engineering
df_train['SMA_10'] = df_train['y'].rolling(window=10).mean()
df_train['SMA_30'] = df_train['y'].rolling(window=30).mean()
df_train['EMA_10'] = df_train['y'].ewm(span=10, adjust=False).mean()
df_train['EMA_30'] = df_train['y'].ewm(span=30, adjust=False).mean()

# Add date-based features
df_train['day'] = df_train['ds'].dt.day
df_train['month'] = df_train['ds'].dt.month
df_train['year'] = df_train['ds'].dt.year

# Drop rows with NaN values from rolling means
df_train = df_train.dropna()

# Predicting future data: Create future dataset (5 days ahead starting from today)
last_known_data = df_train.iloc[-1]
future_dates = pd.date_range(TODAY_DATE, periods=period, freq='D')
future_features = generate_future_features(last_known_data, future_dates)

# Ensure future features have the same column order as expected by the model
X_train_columns = ['SMA_10', 'SMA_30', 'EMA_10', 'EMA_30', 'day', 'month', 'year']
future_features = future_features[X_train_columns]

# Predict future prices using the pre-trained Random Forest model
future_close = best_rf.predict(future_features)

# Create a DataFrame for the predictions
future_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_close})
future_df.set_index('Date', inplace=True)

# Display the forecast data
st.subheader('Bitcoin Price Forecast (Next 5 Days)')
st.dataframe(future_df)

# Create a line plot of historical data and predictions
fig = go.Figure()

# Add historical data
fig.add_trace(go.Scatter(x=df_train['ds'], y=df_train['y'], mode='lines', name='Historical Price'))

# Add predictions
fig.add_trace(go.Scatter(x=future_df.index, y=future_df['Predicted Close'], mode='lines+markers', name='Predicted Price'))

# Update layout
fig.update_layout(title='Bitcoin Price: Historical and Predicted',
                  xaxis_title='Date',
                  yaxis_title='Price (USD)',
                  legend_title='Data Type',
                  hovermode='x')

# Display the plot
st.plotly_chart(fig, use_container_width=True)

# Display additional information
st.subheader('Additional Information')
st.write(f"Data range: {START} to {TODAY}")
st.write(f"Last known price: ${df_train['y'].iloc[-1]:.2f}")
st.write(f"Predicted price on {future_df.index[-1].date()}: ${future_df['Predicted Close'].iloc[-1]:.2f}")

# Disclaimer
st.caption("Disclaimer: This prediction is based on historical data and should not be used as financial advice. Cryptocurrency markets are highly volatile and unpredictable.")
