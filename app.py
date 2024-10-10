import requests
import pickle
import streamlit as st
import pandas as pd
from datetime import date, timedelta
import yfinance as yf

# Google Drive link for the model
MODEL_URL = "https://drive.google.com/uc?export=download&id=1kTRfX5_r2cFDWiJwCUNtREavZgLP18dA"


# Function to download the model from Google Drive
@st.cache_resource
def load_model(url):
    response = requests.get(url)
    with open("bitcoin_rf_model_with_moving_avg.pkl", "wb") as file:
        file.write(response.content)
    
    with open("bitcoin_rf_model_with_moving_avg.pkl", "rb") as file:
        model = pickle.load(file)
    
    return model

# Load the pre-trained model
model = load_model(MODEL_URL)

# Constants
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
selected_stock = 'BTC-USD'
period = 5  # Predicting for the next 5 days

# Function to load stock/crypto data from Yahoo Finance
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Load the data
data = load_data(selected_stock)

# Prepare the data for predictions
df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})

# Feature Engineering
df_train['SMA_10'] = df_train['y'].rolling(window=10).mean()  # 10-day Simple Moving Average
df_train['SMA_30'] = df_train['y'].rolling(window=30).mean()  # 30-day Simple Moving Average
df_train['EMA_10'] = df_train['y'].ewm(span=10, adjust=False).mean()  # 10-day Exponential Moving Average
df_train['EMA_30'] = df_train['y'].ewm(span=30, adjust=False).mean()  # 30-day Exponential Moving Average

df_train['day'] = df_train['ds'].dt.day
df_train['month'] = df_train['ds'].dt.month
df_train['year'] = df_train['ds'].dt.year

# Drop rows with NaN values
df_train = df_train.dropna()

# Ensure feature order matches the training data
features_order = ['SMA_10', 'SMA_30', 'EMA_10', 'EMA_30', 'day', 'month', 'year']

# Streamlit UI
st.title("Bitcoin Price Prediction (Next 5 Days)")

# Show raw data (last 5 rows)
st.subheader("Raw Data")
st.write(data.tail())

# Step 1: Check if there's a gap between the last available date and today
last_available_date = df_train['ds'].max()

# Step 2: Fill missing dates between the last available date and today
if last_available_date < pd.Timestamp(TODAY):
    # Create a date range between the last available date and today
    missing_dates = pd.date_range(last_available_date + timedelta(days=1), pd.Timestamp(TODAY), freq='D')

    # Fill missing dates using the last known values for the moving averages
    last_row = df_train.tail(1)
    missing_data = pd.DataFrame({
        'ds': missing_dates,
        'y': last_row['y'].values[0],  # Use last known price as a placeholder
        'SMA_10': last_row['SMA_10'].values[0],
        'SMA_30': last_row['SMA_30'].values[0],
        'EMA_10': last_row['EMA_10'].values[0],
        'EMA_30': last_row['EMA_30'].values[0],
        'day': [d.day for d in missing_dates],
        'month': [d.month for d in missing_dates],
        'year': [d.year for d in missing_dates]
    })

    # Append the missing data to the existing dataframe
    df_train = pd.concat([df_train, missing_data])

# Step 3: Prepare future features for prediction
today = pd.Timestamp(TODAY)

# Generate future dates starting from today (current date)
future_dates = pd.date_range(today, periods=period, freq='D').tolist()

# Use the most recent feature data for predictions (from the last row of df_train)
last_row = df_train.tail(1)

# Generate new feature data for future dates (moving averages and date features)
future_features = pd.DataFrame({
    'day': [d.day for d in future_dates],
    'month': [d.month for d in future_dates],
    'year': [d.year for d in future_dates],
    'SMA_10': last_row['SMA_10'].values[0],  # Last known SMA_10 value
    'SMA_30': last_row['SMA_30'].values[0],  # Last known SMA_30 value
    'EMA_10': last_row['EMA_10'].values[0],  # Last known EMA_10 value
    'EMA_30': last_row['EMA_30'].values[0]   # Last known EMA_30 value
})

# Ensure future features match the training feature order
future_features = future_features[features_order]

# Predict future prices using the pre-trained model
future_close = model.predict(future_features)

# Create a DataFrame for the predictions
future_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_close})
future_df.set_index('Date', inplace=True)

# Display the forecast data
st.subheader(f"Predicted Bitcoin Prices for the Next {period} Days")
st.write(future_df)

# Plot the predictions
st.subheader("Prediction Plot")
st.line_chart(future_df['Predicted Close'])
