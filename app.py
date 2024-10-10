# Import necessary libraries
from datetime import date, timedelta
import pandas as pd
import yfinance as yf
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

# Prepare the data for predictions (using 'Date' and 'Close' columns)
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Feature Engineering: Adding more features (SMA, EMA, day, month, year)
df_train['SMA_10'] = df_train['y'].rolling(window=10).mean()  # 10-day Simple Moving Average
df_train['SMA_30'] = df_train['y'].rolling(window=30).mean()  # 30-day Simple Moving Average
df_train['EMA_10'] = df_train['y'].ewm(span=10, adjust=False).mean()  # 10-day Exponential Moving Average
df_train['EMA_30'] = df_train['y'].ewm(span=30, adjust=False).mean()  # 30-day Exponential Moving Average

# Add date-based features
df_train['day'] = df_train['ds'].dt.day
df_train['month'] = df_train['ds'].dt.month
df_train['year'] = df_train['ds'].dt.year

# Drop rows with NaN values from rolling means
df_train = df_train.dropna()

# Find the last available date in the data
last_date = df_train['ds'].max()

# If the last date in the data is older than today, fill the gap with missing data
if last_date < TODAY_DATE:
    missing_days = pd.date_range(start=last_date + timedelta(days=1), end=TODAY_DATE)
    
    # Create a DataFrame to fill the gap with placeholder values (use last available values)
    filler_df = pd.DataFrame({
        'ds': missing_days,
        'y': np.nan,  # Placeholder for actual 'Close' values
        'SMA_10': df_train['SMA_10'].iloc[-1],  # Last known SMA_10 value
        'SMA_30': df_train['SMA_30'].iloc[-1],  # Last known SMA_30 value
        'EMA_10': df_train['EMA_10'].iloc[-1],  # Last known EMA_10 value
        'EMA_30': df_train['EMA_30'].iloc[-1],  # Last known EMA_30 value
        'day': missing_days.day,
        'month': missing_days.month,
        'year': missing_days.year
    })
    
    # Append the missing data to the training data
    df_train = pd.concat([df_train, filler_df], ignore_index=True)

# Extract the most recent data (including today) for future prediction
last_row = df_train.tail(1)

# Predicting future data: Create future dataset (5 days ahead starting from today)
future_dates = pd.date_range(TODAY_DATE, periods=period, freq='D').tolist()

# Generate new feature data for future dates (moving averages and date features)
future_features = pd.DataFrame({
    'day': [d.day for d in future_dates],
    'month': [d.month for d in future_dates],
    'year': [d.year for d in future_dates],
    'SMA_10': last_row['SMA_10'].iloc[-1],  # Last known SMA_10 value
    'SMA_30': last_row['SMA_30'].iloc[-1],  # Last known SMA_30 value
    'EMA_10': last_row['EMA_10'].iloc[-1],  # Last known EMA_10 value
    'EMA_30': last_row['EMA_30'].iloc[-1]   # Last known EMA_30 value
})

# Ensure future features have the same column order as X_train
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
