import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import streamlit as st

# Load the diabetes dataset
df = pd.read_csv("diabetes.csv")

# Encoding categorical features
label_encoder_gender = LabelEncoder()
df['gender'] = label_encoder_gender.fit_transform(df['gender'])
label_encoder_smoking = LabelEncoder()
df['smoking_history'] = label_encoder_smoking.fit_transform(df['smoking_history'])

# Splitting features and target variables
X = df[['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']]
y = df['diabetes']

# Split the data into training, validation and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = SVC()
model.fit(X_train, y_train)
model.predict(X_test)

# Streamlit Interface
st.title('Diabetes Prediction')

# Collect user input
st.sidebar.header("Input Features")
gender = st.sidebar.selectbox("Gender", ("Female","Male"))
age = st.sidebar.slider("Age", 0, 100, 25)
hypertension = st.sidebar.selectbox("Hypertension", ["Yes", "No"])
heart_disease = st.sidebar.selectbox("Heart Disease", ["Yes", "No"])
smoking_history = st.sidebar.selectbox("Smoking History", ["No Info", "current", "ever","former","never"])
bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
hba1c_level = st.sidebar.number_input("HbA1c Level", min_value=0.0, max_value=20.0, value=5.0, step=0.1)
blood_glucose_level = st.sidebar.number_input("Blood Glucose Level", min_value=0, max_value=500, value=100, step=1)

# Preprocess user input
gender_encoded = label_encoder_gender.transform([gender])[0]
hypertension = 1 if hypertension == "Yes" else 0
heart_disease = 1 if heart_disease == "Yes" else 0
smoking_encoded = label_encoder_smoking.transform([smoking_history])[0]

# Display result
if st.button("Predict"):
    # Predict diabetes
    input_data = np.array([[gender_encoded, age, hypertension, heart_disease, smoking_encoded, bmi, hba1c_level, blood_glucose_level]])
    prediction = model.predict(input_data)[0]    

    # Displaying results
    st.subheader("Prediction Result")
    if prediction == 1:
        st.write("**Prediction:** The person is likely to have diabetes.")
        st.warning("Consider consulting with a healthcare professional for further advice.")
    else:
        st.write("**Prediction:** The person is not likely to have diabetes.")
        st.success("Maintain a healthy lifestyle to continue minimizing risk.")