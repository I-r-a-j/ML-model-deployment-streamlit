import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the model
model = joblib.load('btc_simple_model.pkl')

st.title('BTC Price Prediction App')

# Determine the number of past values needed for prediction
# You'll need to adjust this based on your specific model
n_past_values = 5  # Example: using 5 past values to predict the next

# Create input fields for past values
past_values = []
for i in range(n_past_values):
    value = st.number_input(f'Enter BTC price {i+1} time step ago', value=0.0, step=0.01)
    past_values.append(value)

# Make prediction
if st.button('Predict Next BTC Price'):
    # Prepare input data
    input_data = np.array(past_values).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(input_data)
    
    st.write(f'The predicted next BTC price is: ${prediction[0]:.2f}')

# Optional: Add a section for plotting past values and prediction
if st.checkbox('Show Plot'):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(range(n_past_values), past_values, label='Past Values')
    ax.plot(n_past_values, prediction, 'ro', label='Prediction')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('BTC Price')
    ax.legend()
    st.pyplot(fig)
