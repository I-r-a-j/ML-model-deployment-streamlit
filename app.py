# Import necessary libraries
import streamlit as st
from pycaret.time_series import load_model, predict_model

# Load the saved time series model (.pkl file)
model_path = '/btc_simple_model.pkl'

# Define the Streamlit app
def main():
    st.title("Time Series Forecasting Web App")
    
    # Forecast horizon input
    forecast_horizon = st.number_input('Enter forecast horizon (number of future periods to predict):', min_value=1, value=10)
    
    # Button to trigger prediction
    if st.button("Make Prediction"):
        # Load the model
        model = load_model(model_path)
        
        # Make predictions
        predictions = predict_model(model, fh=forecast_horizon)
        
        # Display the predictions
        st.subheader("Predictions for the next {} time periods:".format(forecast_horizon))
        st.write(predictions)
        
# Run the app
if __name__ == "__main__":
    main()

