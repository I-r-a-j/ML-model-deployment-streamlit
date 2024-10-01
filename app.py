import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('btc_simple_model.pkl')

st.title('BTC Price Prediction App')

# Assuming this is a time series model, let's create inputs for past values
# Adjust the number of past values based on your model's requirements
n_past_values = 5  # Example: using 5 past values to predict the next

past_values = []
for i in range(n_past_values):
    value = st.number_input(f'BTC price {i+1} time steps ago', value=0.0, step=0.01)
    past_values.append(value)

# Make prediction
if st.button('Predict Next BTC Price'):
    # Prepare input data
    input_data = np.array(past_values).reshape(1, -1)
    
    # Make prediction
    try:
        prediction = model.predict(input_data)
        st.success(f'The predicted next BTC price is: ${prediction[0]:.2f}')
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("If this error persists, the model might need to be loaded differently. Please check the model structure and input requirements.")
