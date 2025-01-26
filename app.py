import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Load the saved model and label encoders
def load_model_and_encoders():
    try:
        # Load model
        with open('diabetes_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load label encoders
        with open('label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        
        return model, label_encoders
    
    except FileNotFoundError:
        st.error("Model files not found. Ensure diabetes_model.pkl and label_encoders.pkl are in the same directory.")
        return None, None

# Streamlit App
def main():
    st.title('Diabetes Prediction App')

    # Load model and encoders
    model, label_encoders = load_model_and_encoders()
    
    if model is not None and label_encoders is not None:
        # Sidebar for input features
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

            # Additional insights
            st.subheader("Risk Factors")
            risk_factors = []
            if hypertension == "Yes":
                risk_factors.append("Hypertension")
            if heart_disease == "Yes":
                risk_factors.append("Heart Disease")
            if bmi > 30:
                risk_factors.append("High BMI")
            if hba1c_level > 6.5:
                risk_factors.append("Elevated HbA1c")
            
            if risk_factors:
                st.write("Identified Risk Factors:", ", ".join(risk_factors))
            else:
                st.write("No additional risk factors identified.")

# Run the app
if __name__ == "__main__":
    main()