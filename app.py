# pip install streamlit yfinance scikit-learn plotly
import streamlit as st
from datetime import date
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go

# Set the start date and today's date
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Streamlit app title
st.title('Stock Forecast App')

# Stock options for the user to select
stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

# Number of years for prediction slider
n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

# Load stock data from yfinance
@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

# Display raw data
st.subheader('Raw data')
st.write(data.tail())

# Function to plot raw stock data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Prepare the data for machine learning model
df_train = data[['Date', 'Close']]
df_train['Date'] = pd.to_datetime(df_train['Date'])
df_train.set_index('Date', inplace=True)

# Create features based on previous closing prices
df_train['Shifted_Close'] = df_train['Close'].shift(1)
df_train = df_train.dropna()

# Create train and test datasets
X = df_train[['Shifted_Close']]  # Using previous close as feature
y = df_train['Close']  # Target variable is the current close price

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initialize and fit the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Create a future dataset for predictions
last_close = df_train['Close'].iloc[-1]
future_dates = pd.date_range(df_train.index[-1], periods=period + 1, freq='D').tolist()

future_close = [last_close]
for i in range(1, period):
    next_close = model.predict([[future_close[-1]]])
    future_close.append(next_close[0])

future_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_close})
future_df.set_index('Date', inplace=True)

# Show the forecast data
st.subheader('Forecast data')
st.write(future_df.tail())

# Plot forecast data
st.write(f'Forecast plot for {n_years} years')
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=future_df.index, y=future_df['Predicted Close'], name='Predicted Close'))
fig1.layout.update(title_text='Forecasted Stock Prices', xaxis_rangeslider_visible=True)
st.plotly_chart(fig1)
