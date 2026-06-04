import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="AI Burnout Predictor", 
    page_icon="🧠", 
    layout="centered"
)

# Title and description
st.title("📊 Employee Burnout Risk Predictor")
st.write("An AI-powered dashboard for HR managers to detect and monitor employee burnout risk levels in real-time.")

# 1. Load ML Artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('burnout_risk_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('selected_features.pkl')
    return model, scaler, features

try:
    model, scaler, selected_features = load_artifacts()
    st.success("✅ AI Model and Scaler loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model artifacts: {e}")
    st.info("Please ensure 'burnout_risk_model.pkl', 'scaler.pkl', and 'selected_features.pkl' are in the same directory.")
    st.stop()

st.divider()
st.subheader("📋 Enter Employee Metrics")

# 2. Dynamic Input Form based on your exact selected_features
input_data = {}

# Layout mapping for a clean 2-column UI
col1, col2 = st.columns(2)

# We check your exact model features and create the sliders/inputs dynamically
for feature in selected_features:
    # Determine which column to place the input
    current_col = col1 if list(selected_features).index(feature) % 2 == 0 else col2
    
    with current_col:
        if feature == 'hours_worked_per_week':
            input_data[feature] = st.number_input("Weekly Work Hours:", min_value=10, max_value=100, value=40)
        elif feature == 'work_life_balance_score':
            input_data[feature] = st.slider("Work-Life Balance Score (0-5):", 0.0, 5.0, 3.0, 0.1)
        elif feature == 'job_satisfaction_score':
            input_data[feature] = st.slider("Job Satisfaction Score (0-5):", 0.0, 5.0, 3.0, 0.1)
        elif feature == 'stress_level_self_report':
            input_data[feature] = st.slider("Self-Reported Stress Level (0-5):", 0.0, 5.0, 2.5, 0.1)
        elif feature == 'anxiety_score':
            input_data[feature] = st.slider("Anxiety Score (0-5):", 0.0, 5.0, 2.0, 0.1)
        elif feature == 'depression_score':
            input_data[feature] = st.slider("Depression Score (0-5):", 0.0, 5.0, 1.5, 0.1)
        else:
            # Fallback for any other continuous feature in your dataset
            input_data[feature] = st.number_input(f"{feature.replace('_', ' ').title()}:", value=0.0)

# 3. Prediction Logic
if st.button("🔍 Predict Burnout Risk", type="primary", use_container_width=True):
    
    # Convert input dict to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Ensure exact feature order expected by the scaler/model
    input_df = input_df[selected_features]
    
    # Apply standard scaling
    input_scaled = scaler.transform(input_df)
    
    # Make prediction and calculate probabilities
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]
    
    # Map classes to their corresponding probabilities dynamically
    prob_dict = dict(zip(model.classes_, prediction_proba))
    
    st.divider()
    st.subheader("🎯 Assessment Result")
    
    # Dynamic and bulletproof conditional logic based on actual prediction string/value
    # Supports both numeric (0, 1, 2) and text ('Low', 'Moderate', 'High') labels
    if prediction in ['High', 2]:
        confidence = prob_dict.get(prediction, max(prediction_proba)) * 100
        st.error(f"🔴 HIGH BURNOUT RISK")
        st.metric(label="Model Confidence", value=f"{confidence:.1f}%")
        st.warning("⚠️ **Urgent Action Needed:** Immediate workload reduction recommended (target: under 50 hours/week). Schedule a 1-on-1 counseling session.")
        
    elif prediction in ['Moderate', 1]:
        confidence = prob_dict.get(prediction, max(prediction_proba)) * 100
        st.warning(f"🟡 MODERATE BURNOUT RISK")
        st.metric(label="Model Confidence", value=f"{confidence:.1f}%")
        st.info("💡 **Recommendation:** Monitor performance indicators, offer flexible working arrangements, and encourage stress management sessions.")
        
    else:
        confidence = prob_dict.get(prediction, max(prediction_proba)) * 100
        st.success(f"🟢 LOW BURNOUT RISK")
        st.metric(label="Model Confidence", value=f"{confidence:.1f}%")
        st.write("✨ **Status Stable:** Employee metrics are within healthy ranges. Maintain current work-life balance standards.")