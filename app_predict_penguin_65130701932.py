import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
@st.cache_data  # Updated caching method
def load_data():
    # Load the seaborn penguin dataset
    penguins = sns.load_dataset('penguins')
    return penguins

# Preprocess the data
def preprocess_data(data):
    # Drop rows with missing values
    data = data.dropna()

    # Encode the 'species' column
    label_encoder = LabelEncoder()
    data['species'] = label_encoder.fit_transform(data['species'])

    # Convert categorical columns to numerical (dummies for 'island' and 'sex')
    data = pd.get_dummies(data, drop_first=True)
    return data

# Function to train the KNN model
def train_knn_model(data):
    # Split data into features and target
    X = data.drop('species', axis=1)
    y = data['species']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train a KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    
    # Predict on test set and evaluate
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return knn, accuracy, X.columns  # Also return the column names

# Main function to run the Streamlit app
def main():
    st.title("Penguin Species Prediction with KNN")

    # Load the data
    data = load_data()
    st.write("Penguin Dataset", data.head())

    # Preprocess the data
    processed_data = preprocess_data(data)

    # Train the KNN model
    model, accuracy, feature_columns = train_knn_model(processed_data)
    
    # Display the model accuracy
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

    # User input for prediction
    st.sidebar.header("User Input for Prediction")
    
    # Create input fields for penguin characteristics
    bill_length = st.sidebar.slider('Bill Length (mm)', float(processed_data['bill_length_mm'].min()), float(processed_data['bill_length_mm'].max()), float(processed_data['bill_length_mm'].mean()))
    bill_depth = st.sidebar.slider('Bill Depth (mm)', float(processed_data['bill_depth_mm'].min()), float(processed_data['bill_depth_mm'].max()), float(processed_data['bill_depth_mm'].mean()))
    flipper_length = st.sidebar.slider('Flipper Length (mm)', float(processed_data['flipper_length_mm'].min()), float(processed_data['flipper_length_mm'].max()), float(processed_data['flipper_length_mm'].mean()))
    body_mass = st.sidebar.slider('Body Mass (g)', float(processed_data['body_mass_g'].min()), float(processed_data['body_mass_g'].max()), float(processed_data['body_mass_g'].mean()))

    # Prepare the input for prediction, including dummy variables for 'island' and 'sex'
    input_data = pd.DataFrame({
        'bill_length_mm': [bill_length],
        'bill_depth_mm': [bill_depth],
        'flipper_length_mm': [flipper_length],
        'body_mass_g': [body_mass],
    })

    # Ensure the columns match by adding missing columns (dummy variables) as 0
    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Make prediction
    species_pred = model.predict(input_data)
    species = ["Adelie", "Chinstrap", "Gentoo"][species_pred[0]]
    
    st.write(f"The predicted penguin species is: {species}")

if __name__ == "__main__":
    main()
