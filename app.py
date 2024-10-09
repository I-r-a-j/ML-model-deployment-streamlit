import requests
import pickle
import streamlit as st
import pandas as pd
from datetime import date
import yfinance as yf  # Import yfinance for downloading stock/crypto data

# Google Drive link for the model
MODEL_URL = "https://drive.google.com/uc?export=download&id=1WXr4WWLJXIwJvOmeSVEINgEI2_IszJw0"

# Function to download the model from Google Drive
@st.cache_resource  # Cache the model to avoid re-downloading
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

# Streamlit UI
st.title("Bitcoin Price Prediction (Next 5 Days)")

# Show raw data (last 5 rows)
st.subheader("Raw Data")
st.write(data.tail())

# Prepare future features for prediction
last_row = df_train.tail(1)
future_dates = pd.date_range(df_train['ds'].iloc[-1], periods=period + 1, freq='D').tolist()

future_features = pd.DataFrame({
    'day': [d.day for d in future_dates],
    'month': [d.month for d in future_dates],
    'year': [d.year for d in future_dates],
    'SMA_10': df_train['SMA_10'].iloc[-1],  # Last known SMA_10 value
    'SMA_30': df_train['SMA_30'].iloc[-1],  # Last known SMA_30 value
    'EMA_10': df_train['EMA_10'].iloc[-1],  # Last known EMA_10 value
    'EMA_30': df_train['EMA_30'].iloc[-1]   # Last known EMA_30 value
})

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
