from datetime import date, timedelta
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
import pickle
import numpy as np

# Set start date for data and today's date
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
TODAY_DATE = pd.to_datetime(TODAY)

# Set Bitcoin symbol for predictions
selected_stock = 'BTC-USD'  # Bitcoin

# Fixed prediction period of 5 days
period = 5

# Function to load stock/crypto data from Yahoo Finance
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Load Bitcoin data up to today's date
data = load_data(selected_stock)

# Display raw data (for debugging)
print("Raw data:")
print(data.tail())

# Load the pre-trained model
model_filename = 'BTC_rf_model_with_moving_avg.pkl'
with open(model_filename, 'rb') as file:
    best_rf = pickle.load(file)

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

# Function to generate future features
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

# Show the forecast data (Next 5 days)
print("Forecast data (Next 5 days):")
print(future_df)
