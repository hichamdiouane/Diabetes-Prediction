import os
import pickle
import numpy as np
import streamlit as st

# Ensure correct file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'diabetes_model.pkl')
ENCODERS_PATH = os.path.join(BASE_DIR, 'label_encoders.pkl')

def load_model_and_encoders():
    try:
        # Detailed file existence check
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found: {MODEL_PATH}")
            st.info("Please ensure you have trained and saved the model.")
            return None, None

        if not os.path.exists(ENCODERS_PATH):
            st.error(f"Encoders file not found: {ENCODERS_PATH}")
            st.info("Please ensure you have saved the label encoders.")
            return None, None

        # Load model
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        
        # Load label encoders
        with open(ENCODERS_PATH, 'rb') as f:
            label_encoders = pickle.load(f)
        
        return model, label_encoders
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Troubleshooting steps:")
        st.info("1. Verify model files are in the same directory as the app")
        st.info("2. Check that model was saved correctly in Jupyter Notebook")
        st.info("3. Ensure pickle compatibility between environments")
        return None, None

def main():
    st.title('Diabetes Prediction App')

    # Attempt to load model
    model, label_encoders = load_model_and_encoders()
    
    # Only proceed if model is successfully loaded
    if model is None:
        st.warning("Cannot proceed without a valid model.")
        return

    # Rest of the Streamlit app remains the same as in previous version
    st.sidebar.header("Patient Information")
    
    # Input features
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    age = st.sidebar.slider("Age", 0, 100, 25)
    hypertension = st.sidebar.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.sidebar.selectbox("Heart Disease", ["No", "Yes"])
    smoking_history = st.sidebar.selectbox("Smoking History", 
        ["No Info", "current", "ever", "former", "never"])
    
    bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
    hba1c_level = st.sidebar.number_input("HbA1c Level", min_value=0.0, max_value=20.0, value=5.0, step=0.1)
    blood_glucose_level = st.sidebar.number_input("Blood Glucose Level", 
        min_value=0, max_value=500, value=100, step=1)

    # Prediction button
    if st.sidebar.button("Predict Diabetes Risk"):
        # Encode categorical variables
        gender_encoded = label_encoders['gender'].transform([gender])[0]
        smoking_encoded = label_encoders['smoking_history'].transform([smoking_history])[0]
        
        # Convert boolean selections to numeric
        hypertension_num = 1 if hypertension == "Yes" else 0
        heart_disease_num = 1 if heart_disease == "Yes" else 0

        # Prepare input data
        input_data = np.array([[
            gender_encoded, age, hypertension_num, heart_disease_num, 
            smoking_encoded, bmi, hba1c_level, blood_glucose_level
        ]])

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Display results
        st.header("Prediction Results")
        if prediction == 1:
            st.error("High Risk of Diabetes")
            st.warning("Consider consulting a healthcare professional for further evaluation.")
        else:
            st.success("Low Risk of Diabetes")
            st.info("Maintain a healthy lifestyle to minimize risk.")

# Run the app
if __name__ == "__main__":
    main()