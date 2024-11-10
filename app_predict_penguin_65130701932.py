import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Disable the warning for using st.cache
st.set_option('deprecation.showfileUploaderEncoding', False)

# Configure page to make it lighter
st.set_page_config(
    page_title="Penguin Prediction",
    page_icon="🐧",
    layout="centered",
    initial_sidebar_state="collapsed"  # Collapse sidebar by default
)

# Optimize model loading with proper caching
@st.cache_resource  # Using cache_resource for model loading to prevent reloading during each interaction
def load_model():
    try:
        with open('model_penguin_65130701932.pkl', 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error("Model file not found!")
        return None

# Load model only once at startup
model_data = load_model()

if model_data is not None:
    model, species_encoder, island_encoder, sex_encoder = model_data
    
    # Streamlined UI
    st.title("🐧 Penguin Prediction")
    
    # Use columns for better organization and faster rendering
    col1, col2 = st.columns(2)
    
    with col1:
        species = st.selectbox("Species", species_encoder.classes_, key='species')
        island = st.selectbox("Island", island_encoder.classes_, key='island')
        sex = st.selectbox("Sex", sex_encoder.classes_, key='sex')
    
    with col2:
        bill_length_mm = st.number_input("Bill Length (mm)", value=40.0, min_value=30.0, max_value=100.0, step=0.1)
        bill_depth_mm = st.number_input("Bill Depth (mm)", value=15.0, min_value=10.0, max_value=60.0, step=0.1)
        flipper_length_mm = st.number_input("Flipper Length (mm)", value=200, min_value=150, max_value=250, step=1)
        body_mass_g = st.number_input("Body Mass (g)", value=4000, min_value=2000, max_value=7000, step=10)

    # Streamlined prediction process
    if st.button("Predict", type="primary", use_container_width=True):
        # Create input data more efficiently
        input_data = pd.DataFrame({
            "bill_length_mm": [bill_length_mm],
            "bill_depth_mm": [bill_depth_mm],
            "flipper_length_mm": [flipper_length_mm],
            "body_mass_g": [body_mass_g],
            "species": species_encoder.transform([species]),
            "island": island_encoder.transform([island]),
            "sex": sex_encoder.transform([sex])
        })
        
        # Make prediction
        prediction = model.predict(input_data)
        predicted_species = species_encoder.inverse_transform(prediction)[0]
        
        # Display result efficiently
        st.success(f"Predicted Species: {predicted_species}")
        
        # Optional: Display compact summary
        with st.expander("View Input Summary"):
            st.write(input_data.iloc[0])

else:
    st.error("Failed to load the model. Please check if the model file exists.")
