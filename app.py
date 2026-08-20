import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved assets
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
encoders = joblib.load('encoders.pkl')
feature_columns = joblib.load('feature_columns.pkl')

st.set_page_config(page_title="HR Attrition Predictor", layout="centered")
st.title("💼 Employee Attrition Risk Assessment")
st.write("Adjust employee parameters to evaluate probability of departure.")

# Input Form
with st.form("employee_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 18, 65, 30)
        monthly_income = st.number_input("Monthly Income ($)", 1000, 20000, 5000)
        total_working_years = st.slider("Total Working Years", 0, 40, 5)
        overtime = st.selectbox("OverTime", ["No", "Yes"])
        job_role = st.selectbox("Job Role", encoders['JobRole'].classes_)
        
    with col2:
        distance_from_home = st.slider("Distance From Home (km)", 1, 50, 10)
        years_at_company = st.slider("Years At Company", 0, 40, 3)
        environment_satisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
        job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
        marital_status = st.selectbox("Marital Status", encoders['MaritalStatus'].classes_)

    submit_button = st.form_submit_button("Predict Attrition Risk")

if submit_button:
    # Build default dataframe matching input feature schema
    input_data = {col: 0 for col in feature_columns}
    
    # Fill in captured inputs
    input_data['Age'] = age
    input_data['MonthlyIncome'] = monthly_income
    input_data['TotalWorkingYears'] = total_working_years
    input_data['DistanceFromHome'] = distance_from_home
    input_data['YearsAtCompany'] = years_at_company
    input_data['EnvironmentSatisfaction'] = environment_satisfaction
    input_data['JobSatisfaction'] = job_satisfaction
    
    # Encode categorical fields
    input_data['OverTime'] = encoders['OverTime'].transform([overtime])[0]
    input_data['JobRole'] = encoders['JobRole'].transform([job_role])[0]
    input_data['MaritalStatus'] = encoders['MaritalStatus'].transform([marital_status])[0]

    # Convert to DataFrame and Scale
    input_df = pd.DataFrame([input_data])
    scaled_data = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1] * 100

    st.markdown("---")
    st.subheader("Results")
    if prediction == 1:
        st.error(f"⚠️ **High Attrition Risk:** {probability:.1f}% chance of leaving.")
    else:
        st.success(f"✅ **Low Attrition Risk:** {probability:.1f}% chance of leaving.")