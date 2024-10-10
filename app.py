# Import necessary libraries
from datetime import date, timedelta
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import pickle
import numpy as np

# Set start date for data and today's date
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Set Bitcoin symbol for training
selected_stock = 'BTC-USD'  # Bitcoin

# Fixed prediction period of 5 days
period = 5

# Function to load stock/crypto data from Yahoo Finance
def load_data(ticker):
    # Fetch data up to today
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Load Bitcoin data (ensure data is up-to-date)
data = load_data(selected_stock)

# Prepare data for forecasting (using 'Date' and 'Close' columns)
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

# Train-test split (only if you want to retrain)
X = df_train[['SMA_10', 'SMA_30', 'EMA_10', 'EMA_30', 'day', 'month', 'year']]
y = df_train['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Random Forest model (retraining or using existing)
# rf = RandomForestRegressor() 
# You don't need to retrain if you're loading the existing model
# rf.fit(X_train, y_train)

# Load the pre-trained model
model_filename = 'BTC_rf_model_with_moving_avg.pkl'
with open(model_filename, 'rb') as file:
    best_rf = pickle.load(file)

# Predicting future data: Create future dataset (5 days ahead from today)
last_row = df_train.tail(1)

# Generate future dates starting from today, not the last date in the data
future_dates = pd.date_range(TODAY, periods=period, freq='D').tolist()

# Generate new feature data for future dates (moving averages and date features)
# We use the latest known SMA, EMA values for prediction.
future_features = pd.DataFrame({
    'day': [d.day for d in future_dates],
    'month': [d.month for d in future_dates],
    'year': [d.year for d in future_dates],
    'SMA_10': df_train['SMA_10'].iloc[-1],  # Last known SMA_10 value
    'SMA_30': df_train['SMA_30'].iloc[-1],  # Last known SMA_30 value
    'EMA_10': df_train['EMA_10'].iloc[-1],  # Last known EMA_10 value
    'EMA_30': df_train['EMA_30'].iloc[-1]   # Last known EMA_30 value
})

# Ensure future features have the same column order as X_train
future_features = future_features[X_train.columns]

# Predict future prices using the best RF model
future_close = best_rf.predict(future_features)

# Create a DataFrame for the predictions
future_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_close})
future_df.set_index('Date', inplace=True)

# Show the forecast data
print("Forecast data (Next 5 days):")
print(future_df.tail())
