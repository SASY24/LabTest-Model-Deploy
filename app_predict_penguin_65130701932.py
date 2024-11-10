import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Configure the Streamlit page
st.set_page_config(
    page_title="Penguin Species Prediction",
    page_icon="🐧",
    layout="centered"
)

# Create a function to load the model and encoders
@st.cache_data  # Updated from @st.cache which is deprecated
def load_model():
    try:
        with open('model_penguin_65130701932.pkl', 'rb') as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'model_penguin_65130701932.pkl' exists in the same directory.")
        return None

# Load the model and encoders
try:
    model, species_encoder, island_encoder, sex_encoder = load_model()

    # Main app UI
    st.title("🐧 Penguin Species Prediction")
    st.write("Enter the penguin's characteristics to predict its species.")

    # Create two columns for input fields
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Categorical Features")
        species = st.selectbox(
            "Species",
            options=species_encoder.classes_,
            help="Select the penguin's species"
        )
        
        island = st.selectbox(
            "Island",
            options=island_encoder.classes_,
            help="Select the island where the penguin was observed"
        )
        
        sex = st.selectbox(
            "Sex",
            options=sex_encoder.classes_,
            help="Select the penguin's sex"
        )

    with col2:
        st.subheader("Measurements")
        bill_length_mm = st.number_input(
            "Bill Length (mm)",
            min_value=30.0,
            max_value=100.0,
            step=0.1,
            help="Enter the bill length in millimeters"
        )
        
        bill_depth_mm = st.number_input(
            "Bill Depth (mm)",
            min_value=10.0,
            max_value=60.0,
            step=0.1,
            help="Enter the bill depth in millimeters"
        )
        
        flipper_length_mm = st.number_input(
            "Flipper Length (mm)",
            min_value=150,
            max_value=250,
            step=1,
            help="Enter the flipper length in millimeters"
        )
        
        body_mass_g = st.number_input(
            "Body Mass (g)",
            min_value=2000,
            max_value=7000,
            step=10,
            help="Enter the body mass in grams"
        )

    # Add a prediction button with custom styling
    st.markdown("---")
    predict_button = st.button("Predict Species", type="primary")

    if predict_button:
        try:
            # Create input array for prediction
            input_data = pd.DataFrame([[
                bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g
            ]], columns=["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"])
            
            # Add encoded categorical variables
            input_data["species"] = species_encoder.transform([species])
            input_data["island"] = island_encoder.transform([island])
            input_data["sex"] = sex_encoder.transform([sex])

            # Make prediction
            prediction = model.predict(input_data)
            predicted_species = species_encoder.inverse_transform(prediction)[0]

            # Display result with styling
            st.success(f"Predicted Penguin Species: **{predicted_species}**")
            
            # Display input summary
            st.subheader("Input Summary")
            summary_df = pd.DataFrame({
                'Feature': ['Island', 'Sex', 'Bill Length', 'Bill Depth', 'Flipper Length', 'Body Mass'],
                'Value': [island, sex, f"{bill_length_mm} mm", f"{bill_depth_mm} mm", 
                         f"{flipper_length_mm} mm", f"{body_mass_g} g"]
            })
            st.table(summary_df)

        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")

except Exception as e:
    st.error(f"Failed to initialize the application: {str(e)}")

# Add footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
