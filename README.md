# Diabetes Prediction Project

## Overview
This project implements a machine learning solution to predict diabetes risk in patients based on various health metrics. The model is trained on the Diabetes dataset from Kaggle and is deployed as a web application using Streamlit for easy user interaction.

## Screenshots

### Web Application Interface
![Streamlit App Main Interface](/screenshots/main_interface.jpeg)

### Prediction Results
![Diabetes Low Risk Prediction](/screenshots/prediction_result1.jpeg)
![Diabetes High Risk Prediction](/screenshots/prediction_result2.jpeg)

## Project Structure
```
├── screenshots/               # Directory for application screenshots
│   ├── main_interface.png
│   └── prediction_result.png
├── Diabetes Prediction.ipynb   # Main notebook with data analysis and model training
├── README.md                  # Project documentation
├── app.py                     # Streamlit web application
└── diabetes.csv              # Dataset file
```

## Dataset
The dataset used in this project is sourced from Kaggle and includes the following features:
- gender
- age
- hypertension
- heart_disease
- smoking_history
- bmi
- HbA1c_level
- blood_glucose_level
- diabetes (Target variable: 1 for diabetes, 0 for no diabetes)

## Machine Learning Models
We implemented and compared several classification algorithms:
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- K-Nearest Neighbors (KNN)

Each model was evaluated using multiple metrics:
- Accuracy
- Precision

## Web Application
The `app.py` file contains a Streamlit web application that:
- Provides a user-friendly interface for inputting patient data
- Uses the best-performing model to make predictions
- Displays the prediction result

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/diabetes-prediction.git
cd Diabetes Prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

## Usage

### Jupyter Notebook
- Open `Diabetes Prediction.ipynb` to:
  - View the data analysis process
  - See model training and evaluation
  - Access performance metrics and visualizations

### Web Application
1. Start the Streamlit app
2. Enter patient health metrics in the provided fields
3. Click "Predict" to see the diabetes risk assessment

## Dependencies
- Python 3.8+
- pandas
- numpy
- scikit-learn
- streamlit
- matplotlib

## Future Improvements
- Implement feature importance analysis
- Add more advanced models (Random Forest, XGBoost)
- Enhance the web interface with additional visualizations
- Add model explanation using SHAP or LIME
- Implement regular model retraining pipeline
