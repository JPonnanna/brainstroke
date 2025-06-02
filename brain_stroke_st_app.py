import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load models
rf_model = pickle.load(open('model1.pkl', 'rb'))
import xgboost as xgb
xgb_model = xgb.Booster()
xgb_model.load_model("xgb_model.json")

meta_model = pickle.load(open('model3.pkl', 'rb'))

st.title("🧠 Brain Stroke Risk Predictor")

# Layout
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.slider("Age", 0, 120, 30)
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0, value=100.0)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
    stress_level = st.selectbox("Stress Level", ["low", "medium", "high"])

with col2:
    exercise_frequency = st.selectbox("Exercise Frequency", ["none", "1-2x/wk", "3-5x/wk", "daily"])
    alcohol_intake = st.selectbox("Alcohol Intake", ["none", "occasional", "regular"])
    diet_type = st.selectbox("Diet Type", ["healthy", "average", "unhealthy"])
    sleep_quality = st.selectbox("Sleep Quality", ["poor", "average", "good"])
    stroke_family_history = st.selectbox("Stroke Family History", ["No", "Yes"])
    salt_intake = st.selectbox("Salt Intake", ["low", "moderate", "high"])
    systolic_bp = st.number_input("Systolic BP", min_value=90.0, max_value=200.0, value=120.0)
    diastolic_bp = st.number_input("Diastolic BP", min_value=60.0, max_value=130.0, value=80.0)
    ldl_cholesterol = st.number_input("LDL Cholesterol", min_value=50.0, max_value=300.0, value=120.0)
    hdl_cholesterol = st.number_input("HDL Cholesterol", min_value=10.0, max_value=100.0, value=40.0)

# Convert binary to 0/1
hypertension = 1 if hypertension == "Yes" else 0
heart_disease = 1 if heart_disease == "Yes" else 0
stroke_family_history = 1 if stroke_family_history == "Yes" else 0
#input_data=pd.DataFrame()
# Final prediction on submit
if st.button("Predict Stroke Risk"):
    input_data = pd.DataFrame({
        "gender": [gender],
        "age": [age],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],
        "ever_married": [ever_married],
        "work_type": [work_type],
        "Residence_type": [residence_type],
        "avg_glucose_level": [avg_glucose_level],
        "bmi": [bmi],
        "smoking_status": [smoking_status],
        "stress_level": [stress_level],
        "exercise_frequency": [exercise_frequency],
        "alcohol_intake": [alcohol_intake],
        "diet_type": [diet_type],
        "sleep_quality": [sleep_quality],
        "stroke_family_history": [stroke_family_history],
        "salt_intake": [salt_intake],
        "systolic_bp": [systolic_bp],
        "diastolic_bp": [diastolic_bp],
        "LDL_cholesterol": [ldl_cholesterol],
        "HDL_cholesterol": [hdl_cholesterol]
    })
    # Mapping categorical values
    gender_map = {'Male': 1, 'Female': 0}
    yesno_map = {'Yes': 1, 'No': 0}
    residence_map = {'Urban': 1, 'Rural': 0}
    diet_map = {'healthy': 1, 'unhealthy': 0}
    sleep_map = {'poor': 0, 'good': 1}
    salt_map = {'low': 0, 'medium': 1, 'high': 2}
    stress_map = {'low': 0, 'medium': 1, 'high': 2}
    exercise_map = {'never': 0, '1-2x/wk': 1, '3-5x/wk': 2, 'daily': 3}
    alcohol_map = {'never': 0, 'occasional': 1, 'regular': 2}
    smoking_map = {'never smoked': 0, 'formerly smoked': 1, 'smokes': 2}
    # Mapping for work_type
    work_type_map = {
        'Govt_job': 0,
        'Never_worked': 1,
        'Private': 2,
        'Self-employed': 3,
        'Unemployed': 4
    }
    
    # Apply mapping to the work_type column
    input_data['work_type'] = input_data['work_type'].map(work_type_map)
    
        
    # Apply mappings to input
    input_data['gender'] = input_data['gender'].map(gender_map)
    input_data['ever_married'] = input_data['ever_married'].map(yesno_map)
    input_data['Residence_type'] = input_data['Residence_type'].map(residence_map)
    input_data['diet_type'] = input_data['diet_type'].map(diet_map)
    input_data['sleep_quality'] = input_data['sleep_quality'].map(sleep_map)
    input_data['salt_intake'] = input_data['salt_intake'].map(salt_map)
    input_data['stress_level'] = input_data['stress_level'].map(stress_map)
    input_data['exercise_frequency'] = input_data['exercise_frequency'].map(exercise_map)
    input_data['alcohol_intake'] = input_data['alcohol_intake'].map(alcohol_map)
    input_data['smoking_status'] = input_data['smoking_status'].map(smoking_map)
    
    # Convert "Yes"/"No" for binary columns
    # input_data['stroke_family_history'] = input_data['stroke_family_history'].map(yesno_map)
    # input_data['heart_disease'] = input_data['heart_disease'].map(yesno_map)
    # input_data['hypertension'] = input_data['hypertension'].map(yesno_map)
    
# Base predictions
    rf_pred = rf_model.predict_proba(input_data)[:, 1]
    xgb_pred = xgb_model.predict_proba(input_data)[:, 1]
    
    # Meta model prediction
    stacked = np.column_stack((rf_pred, xgb_pred))
    final_pred = meta_model.predict(stacked)[0]
    final_prob = meta_model.predict_proba(stacked)[0][1]
    
    # Display result
    if final_pred == 1:
        st.error(f"⚠️ High Stroke Risk (Probability: {final_prob:.2f})")
    else:
        st.success(f"✅ Low Stroke Risk (Probability: {final_prob:.2f})")
