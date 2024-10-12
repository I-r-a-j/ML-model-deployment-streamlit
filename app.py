# Import necessary libraries
import requests
import pickle
import streamlit as st
from datetime import date
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from plotly import graph_objs as go

# Set start date for data and today's date
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
# Google Drive link for the model
MODEL_URL = "https://drive.google.com/uc?export=download&id=1s7oixIxTfi72ctvnZigUj64XYKAGLqou"

# Set the title of the Streamlit app
st.title('Stock Forecast App')
# Function to download the model
@st.cache_resource  # Caches the model to avoid re-downloading
def download_model(url):
    response = requests.get(url)
    with open("model.pkl", "wb") as file:
        file.write(response.content)

# Available stocks for selection
stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)

# Fixed prediction period of 5 days
period = 5
    return model

# Function to load stock data from Yahoo Finance
@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data
# Load the model
model = download_model(MODEL_URL)

# Load and display stock data
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')
# Function to make predictions (adjust according to your model's input/output)
def predict(data):
    return model.predict(data)

# Display raw data
st.subheader('Raw data')
st.write(data.tail())
# Streamlit UI
st.title("Model Prediction App")

# Plot raw data using Plotly
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
# Input data for prediction (this is just an example; adjust as per your model)
input_data = st.text_area("Enter data for prediction (e.g., CSV format)")

plot_raw_data()
# Prepare data for forecasting (using 'Date' and 'Close' columns)
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
# Train a simple linear regression model for forecasting
model = LinearRegression()
model.fit(df_train.index.values.reshape(-1, 1), df_train['y'])
# Create future dataset for predictions (5 days ahead)
last_index = df_train.index[-1]
future_dates = pd.date_range(df_train['ds'].iloc[-1], periods=period + 1, freq='D').tolist()
future_close = [df_train['y'].iloc[-1]]  # Start with the last known close price
# Recursive prediction loop for future prices (5 days ahead)
for i in range(1, len(future_dates)):
    next_close = model.predict([[last_index + i]])  # Predict next day's close based on linear regression
    future_close.append(next_close[0])
# Ensure future_dates and future_close have the same length before creating a DataFrame
if len(future_dates) == len(future_close):
    future_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_close})
    future_df.set_index('Date', inplace=True)
    # Show the forecast data
    st.subheader('Forecast data (Next 5 days)')
    st.write(future_df.tail())
    # Plot forecast data
    st.write(f'Forecast plot for the next 5 days')
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=future_df.index, y=future_df['Predicted Close'], name='Predicted Close'))
    fig1.layout.update(title_text='Forecasted Stock Prices (5 days)', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig1)
else:
    st.error("Error: Future dates and predicted close values have mismatched lengths.")
if input_data:
    try:
        # Convert input data into a format usable by your model
        df = pd.read_csv(pd.compat.StringIO(input_data))
        prediction = predict(df)
        st.write("Prediction:", prediction)
    except Exception as e:
        st.write(f"Error: {e}")
