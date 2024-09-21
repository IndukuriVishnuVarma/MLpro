import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.base import BaseEstimator  # Importing to check if model is a valid model

# Path to model and dataset
model_path = r'C:\Users\ivish\car_price_prediction_model_cleaned.pkl'
data_path = r'C:\Users\ivish\OneDrive\Desktop\car_prices.csv'

# Load dataset (for dropdowns and further exploration if needed)
try:
    data = pd.read_csv(data_path)
    st.write("Dataset loaded successfully.")
except FileNotFoundError:
    st.error(f"Dataset not found at path: {data_path}")
    data = None

# Load the pre-trained model
try:
    model = joblib.load(model_path)
    if not isinstance(model, BaseEstimator):
        st.error("Loaded object is not a valid model. Please check the model file.")
except FileNotFoundError:
    st.error(f"Model file not found. Please check the path: {model_path}")
    model = None

# Only proceed if both model and data are loaded successfully
if data is not None and model is not None:

    # Get unique values for 'make' and 'model' from the dataset for dropdowns
    unique_makes = data['make'].unique()
    unique_models = data['model'].unique()

    # Title of the app
    st.title('Car Price Prediction App')

    st.write("""
    This app predicts the selling price of a used car based on features such as year, make, model, odometer reading, and condition.
    """)

    # Sidebar for user input features
    st.sidebar.header('Input Car Features')

    # Input fields
    year = st.sidebar.slider('Year of the Car', min_value=int(data['year'].min()), max_value=int(data['year'].max()), value=2018)
    make = st.sidebar.selectbox('Make of the Car', unique_makes)
    model_car = st.sidebar.selectbox('Model of the Car', unique_models)
    odometer = st.sidebar.number_input('Odometer Reading (in miles)', min_value=0, value=30000)
    condition = st.sidebar.slider('Condition (0-100)', min_value=0, max_value=100, value=50)

    # Convert input into DataFrame
    input_data = pd.DataFrame({
        'year': [year],
        'make': [make],
        'model': [model_car],
        'odometer': [odometer],
        'condition': [condition]
    })

    st.write('### Car Features Provided by User:')
    st.write(input_data)

    # Predict button
    if st.button('Predict Price'):
        try:
            prediction = model.predict(input_data)
            st.subheader(f'Estimated Selling Price: ${prediction[0]:,.2f}')
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")

    st.write('---')
    st.write('Developed by VISHNU VARMA')
else:
    st.error("Unable to load either the model or dataset. Please check your file paths.")
