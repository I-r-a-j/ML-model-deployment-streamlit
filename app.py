import streamlit as st
import pandas as pd
import yfinance as yf
import joblib
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Load the saved model
@st.cache_resource
def load_model():
    return joblib.load('bitcoin_price_model2.pkl')

# Fetch the latest Bitcoin data
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_bitcoin_data():
    btc = yf.Ticker("BTC-USD")
    data = btc.history(period="60d")  # Fetch 60 days of data
    return data

# Prepare features for prediction
def prepare_features(data):
    features = data[['Open', 'High', 'Low', 'Volume']].copy()
    features['SMA_7'] = data['Close'].rolling(window=7).mean()
    features['SMA_30'] = data['Close'].rolling(window=30).mean()
    features['EMA_7'] = data['Close'].ewm(span=7, adjust=False).mean()
    features['EMA_30'] = data['Close'].ewm(span=30, adjust=False).mean()
    features['Day'] = data.index.day
    features['Month'] = data.index.month
    features['Year'] = data.index.year
    return features

# Make predictions for the next 5 days
def predict_next_5_days(model, scaler, last_known_values):
    next_5_days = pd.date_range(start=datetime.now().date() + timedelta(days=1), periods=5)
    next_5_days_features = pd.DataFrame(index=next_5_days, columns=['Open', 'High', 'Low', 'Volume', 'SMA_7', 'SMA_30', 'EMA_7', 'EMA_30', 'Day', 'Month', 'Year'])

    next_5_days_features['Day'] = next_5_days.day
    next_5_days_features['Month'] = next_5_days.month
    next_5_days_features['Year'] = next_5_days.year

    for col in next_5_days_features.columns:
        if col not in ['Day', 'Month', 'Year']:
            next_5_days_features[col] = last_known_values[col]

    next_5_days_scaled = scaler.transform(next_5_days_features)
    predictions = model.predict(next_5_days_scaled)

    return pd.Series(predictions, index=next_5_days, name='Predicted Close')

# Main Streamlit app
def main():
    st.title('Bitcoin Price Predictor')

    # Load model and data
    model_data = load_model()
    bitcoin_data = fetch_bitcoin_data()

    # Display the last 10 days of Bitcoin data
    st.subheader('Last 10 Days of Bitcoin Data')
    st.dataframe(bitcoin_data.tail(10))

    # Prepare features and make predictions
    features = prepare_features(bitcoin_data)
    last_known_values = features.iloc[-1]
    predictions = predict_next_5_days(model_data['model'], model_data['scaler'], last_known_values)

    # Display predictions
    st.subheader(f'Predictions for the Next 5 Days (as of {datetime.now().strftime("%Y-%m-%d %H:%M:%S")})')
    st.dataframe(predictions)

    # Create a line plot of historical prices and predictions
    fig = go.Figure()

    # Historical prices
    fig.add_trace(go.Scatter(x=bitcoin_data.index, y=bitcoin_data['Close'],
                             mode='lines',
                             name='Historical Close Price'))

    # Predictions
    fig.add_trace(go.Scatter(x=predictions.index, y=predictions,
                             mode='lines+markers',
                             name='Predicted Close Price'))

    fig.update_layout(title='Bitcoin Price: Historical and Predicted',
                      xaxis_title='Date',
                      yaxis_title='Price (USD)')

    st.plotly_chart(fig)

    # Display last update time
    st.text(f'Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

if __name__ == "__main__":
    main()
