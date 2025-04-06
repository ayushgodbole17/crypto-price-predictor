import streamlit as st
import requests

# Set the API base URL (replace with your EC2 IP or domain)
API_BASE_URL = "http://34.207.219.227"

st.title("Crypto Price Predictor Dashboard")

# Section: Home Status
st.header("Application Status")
try:
    response = requests.get(f"{API_BASE_URL}/")
    if response.status_code == 200:
        st.success("API is up!")
        st.write(response.json())
    else:
        st.error("API did not respond as expected.")
except Exception as e:
    st.error(f"Error accessing API: {e}")

# Section: Prediction
st.header("Make a Prediction")
model_type = st.selectbox("Select Model Type", ["linear", "rf"])
if st.button("Get Prediction"):
    try:
        response = requests.get(f"{API_BASE_URL}/predict", params={"model_type": model_type})
        if response.status_code == 200:
            data = response.json()
            st.write("Prediction:", data.get("prediction"))
            st.write("Model Type:", data.get("model_type"))
        else:
            st.error("Prediction API error: " + response.text)
    except Exception as e:
        st.error(f"Error: {e}")

# Section: Retrain Model
st.header("Retrain Model")
if st.button("Retrain Model"):
    try:
        response = requests.post(f"{API_BASE_URL}/retrain")
        if response.status_code == 200:
            data = response.json()
            st.success("Model retrained successfully!")
            st.write("Training RMSE:", data.get("training_rmse"))
            st.write("Testing RMSE:", data.get("testing_rmse"))
        else:
            st.error("Retraining API error: " + response.text)
    except Exception as e:
        st.error(f"Error: {e}")
