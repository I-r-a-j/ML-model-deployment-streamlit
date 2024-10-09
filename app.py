import requests
import pickle
import streamlit as st
import pandas as pd

# Google Drive link for the model
MODEL_URL = "https://drive.google.com/uc?export=download&id=1WXr4WWLJXIwJvOmeSVEINgEI2_IszJw0"

# Function to download the model
@st.cache_resource  # Caches the model to avoid re-downloading
def download_model(url):
    response = requests.get(url)
    with open("model.pkl", "wb") as file:
        file.write(response.content)

    with open("model.pkl", "rb") as file:
        model = pickle.load(file)

    return model

# Load the model
model = download_model(MODEL_URL)

# Function to make predictions (adjust according to your model's input/output)
def predict(data):
    return model.predict(data)

# Streamlit UI
st.title("Model Prediction App")

# Input data for prediction (this is just an example; adjust as per your model)
input_data = st.text_area("Enter data for prediction (e.g., CSV format)")

if input_data:
    try:
        # Convert input data into a format usable by your model
        df = pd.read_csv(pd.compat.StringIO(input_data))
        prediction = predict(df)
        st.write("Prediction:", prediction)
    except Exception as e:
        st.write(f"Error: {e}")
