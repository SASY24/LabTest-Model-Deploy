import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Disable Streamlit warning for using file uploader encoding
st.set_option('deprecation.showfileUploaderEncoding', False)

# Configure Streamlit page
st.set_page_config(
    page_title="Penguin Prediction",
    page_icon="🐧",
    layout="centered",
    initial_sidebar_state="collapsed"  # Collapse sidebar by default
)

# Function to load the model
@st.cache_resource
def load_model():
    with open('model_penguin.pkl', 'rb') as file:
        return pickle.load(file)

# Load the model and encoders
model, species_encoder, island_encoder, sex_encoder = load_model()

# Streamlit UI
st.title("🐧 Penguin Prediction")

# Input fields for user input
species = st.selectbox("Species", species_encoder.classes_)
island = st.selectbox("Island", island_encoder.classes_)
sex = st.selectbox("Sex", sex_encoder.classes_)
bill_length_mm = st.number_input("Bill Length (mm)", min_value=30.0, max_value=100.0, step=0.1, value=40.0)
bill_depth_mm = st.number_input("Bill Depth (mm)", min_value=10.0, max_value=60.0, step=0.1, value=15.0)
flipper_length_mm = st.number_input("Flipper Length (mm)", min_value=150, max_value=250, step=1, value=200)
body_mass_g = st.number_input("Body Mass (g)", min_value=2000, max_value=7000, step=10, value=4000)

# Predict button
if st.button("Predict"):
    # Prepare the input data
    input_data = pd.DataFrame({
        "bill_length_mm": [bill_length_mm],
        "bill_depth_mm": [bill_depth_mm],
        "flipper_length_mm": [flipper_length_mm],
        "body_mass_g": [body_mass_g],
        "species": species_encoder.transform([species]),
        "island": island_encoder.transform([island]),
        "sex": sex_encoder.transform([sex])
    })
    
    # Make the prediction
    prediction = model.predict(input_data)
    predicted_species = species_encoder.inverse_transform(prediction)[0]

    # Show the result
    st.success(f"Predicted Species: {predicted_species}")
