"""
Heart Disease Prediction System - Streamlit GUI
Interactive web interface for disease prediction, medical reports, and prescriptions
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from prediction import (
    predict_heart_disease, 
    MODEL_LOADED, 
    MODEL_METADATA
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="❤️ Heart Disease Detection System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .positive-prediction {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        font-size: 1.2rem;
    }
    .negative-prediction {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        font-size: 1.2rem;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN PAGE
# ============================================================================
st.title("❤️ Heart Disease Detection & Analysis System")
st.markdown("""
This tool uses a machine learning model to detect heart disease risk and 
generates comprehensive medical reports and prescriptions using AI.
""")

st.markdown("---")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📋 Patient Information", "🔍 Prediction Results", "📊 Reports & Recommendations"])

# ============================================================================
# TAB 1: PATIENT INFORMATION INPUT
# ============================================================================
with tab1:
    st.subheader("Enter Patient Health Information")
    
    # Create columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Demographics")
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=55, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"], index=0)
    
    with col2:
        st.markdown("### Vital Signs")
        blood_pressure = st.number_input("Blood Pressure (Systolic, mmHg)", min_value=80, max_value=250, value=140, step=5)
        cholesterol = st.number_input("Cholesterol Level (mg/dL)", min_value=100, max_value=400, value=240, step=10)
    
    with col3:
        st.markdown("### Body Metrics")
        bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=28.5, step=0.1)
        sleep_hours = st.number_input("Sleep Hours (per night)", min_value=0.0, max_value=12.0, value=7.0, step=0.5)
    
    st.markdown("---")
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown("### Laboratory Tests")
        triglyceride_level = st.number_input("Triglyceride Level (mg/dL)", min_value=0, max_value=500, value=150, step=10)
        crp_level = st.number_input("CRP Level (C-Reactive Protein, mg/L)", min_value=0.0, max_value=20.0, value=3.0, step=0.1)
    
    with col5:
        st.markdown("### Blood Tests")
        fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"], index=0)
        homocysteine_level = st.number_input("Homocysteine Level (μmol/L)", min_value=0.0, max_value=50.0, value=10.0, step=0.5)
    
    with col6:
        st.markdown("### Lifestyle Factors")
        smoking = st.selectbox("Smoking", ["No", "Yes"], index=0)
        exercise_habits = st.selectbox("Exercise Habits", ["Low", "Medium", "High"], index=1)
    
    st.markdown("---")
    
    col7, col8, col9 = st.columns(3)
    
    with col7:
        st.markdown("### Medical History")
        family_heart_disease = st.selectbox("Family History of Heart Disease", ["No", "Yes"], index=0)
        diabetes = st.selectbox("Diabetes", ["No", "Yes"], index=0)
    
    with col8:
        st.markdown("### Cholesterol Profile")
        high_blood_pressure = st.selectbox("High Blood Pressure", ["No", "Yes"], index=0)
        low_hdl_cholesterol = st.selectbox("Low HDL Cholesterol", ["No", "Yes"], index=0)
    
    with col9:
        st.markdown("### Stress & Diet")
        high_ldl_cholesterol = st.selectbox("High LDL Cholesterol", ["No", "Yes"], index=0)
        stress_level = st.selectbox("Stress Level", ["Low", "Medium", "High"], index=1)
    
    st.markdown("---")
    
    col10, col11 = st.columns(2)
    
    with col10:
        alcohol_consumption = st.selectbox("Alcohol Consumption", ["Low", "Medium", "High"], index=0)
    
    with col11:
        sugar_consumption = st.selectbox("Sugar Consumption", ["Low", "Medium", "High"], index=1)
    
    # Prepare patient data dictionary
    patient_info = {
        'age': str(age),
        'gender': gender,
        'blood_pressure': str(blood_pressure),
        'cholesterol': str(cholesterol),
        'bmi': str(bmi),
        'sleep_hours': str(sleep_hours),
        'triglyceride_level': str(triglyceride_level),
        'fasting_blood_sugar': fasting_blood_sugar,
        'crp_level': str(crp_level),
        'homocysteine_level': str(homocysteine_level),
        'smoking': smoking,
        'family_heart_disease': family_heart_disease,
        'diabetes': diabetes,
        'high_blood_pressure': high_blood_pressure,
        'low_hdl_cholesterol': low_hdl_cholesterol,
        'high_ldl_cholesterol': high_ldl_cholesterol,
        'exercise_habits': exercise_habits,
        'stress_level': stress_level,
        'alcohol_consumption': alcohol_consumption,
        'sugar_consumption': sugar_consumption
    }
    
    # Store in session state
    st.session_state.patient_info = patient_info
    
    # Make prediction button
    st.markdown("---")
    col_predict, col_clear = st.columns(2)
    
    with col_predict:
        if st.button("🔍 Generate Prediction & Report", use_container_width=True, key="predict_btn"):
            st.session_state.run_prediction = True
    
    with col_clear:
        if st.button("🔄 Clear Form", use_container_width=True, key="clear_btn"):
            st.session_state.run_prediction = False
            st.rerun()

# ============================================================================
# TAB 2: PREDICTION RESULTS
# ============================================================================
with tab2:
    st.subheader("Prediction Results")
    
    if st.session_state.get('run_prediction', False):
        with st.spinner("⏳ Analyzing patient data and generating prediction..."):
            result = predict_heart_disease(st.session_state.patient_info)
        
        if result:
            st.session_state.prediction_result = result
            
            # Display prediction
            col1, col2 = st.columns(2)
            
            with col1:
                prediction = result['prediction']
                prediction_text = result['prediction_text']
                confidence = result['confidence']
                
                if prediction == 1:
                    st.markdown(f"""
                    <div class="positive-prediction">
                    <strong>⚠️ PREDICTION: {prediction_text}</strong><br>
                    Confidence: {confidence:.1%}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="negative-prediction">
                    <strong>✅ PREDICTION: {prediction_text}</strong><br>
                    Confidence: {confidence:.1%}
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                # Display key metrics
                st.metric("Prediction Confidence", f"{confidence:.2%}")
                risk_level = "High Risk" if confidence > 0.7 and prediction == 1 else "Moderate Risk" if prediction == 1 else "Low Risk"
                st.metric("Risk Level", risk_level)
            
            st.markdown("---")
            
            # Display patient summary
            st.subheader("📋 Patient Summary")
            
            summary_data = {
                'Metric': ['Age', 'Gender', 'Blood Pressure', 'Cholesterol', 'BMI', 'Exercise', 'Smoking', 'Diabetes'],
                'Value': [
                    f"{st.session_state.patient_info['age']} years",
                    st.session_state.patient_info['gender'],
                    f"{st.session_state.patient_info['blood_pressure']} mmHg",
                    f"{st.session_state.patient_info['cholesterol']} mg/dL",
                    st.session_state.patient_info['bmi'],
                    st.session_state.patient_info['exercise_habits'],
                    st.session_state.patient_info['smoking'],
                    st.session_state.patient_info['diabetes']
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        else:
            st.error("❌ Error generating prediction. Please check the input data and try again.")
    
    else:
        st.info("📝 Please fill in patient information in the 'Patient Information' tab and click 'Generate Prediction & Report' to see results.")

# ============================================================================
# TAB 3: REPORTS & RECOMMENDATIONS
# ============================================================================
with tab3:
    st.subheader("Medical Reports & Recommendations")
    
    if st.session_state.get('prediction_result', None):
        result = st.session_state.prediction_result
        
        # Create subtabs for different reports
        report_tab1, report_tab2, report_tab3 = st.tabs(["📄 Medical Report", "💊 Prescriptions", "💡 Suggestions"])
        
        with report_tab1:
            st.markdown("### Comprehensive Medical Report")
            report = result.get('medical_report', 'Report not available')
            st.markdown(report)
            
            # Download button for report
            st.download_button(
                label="📥 Download Medical Report (TXT)",
                data=report,
                file_name="medical_report.txt",
                mime="text/plain"
            )
        
        with report_tab2:
            st.markdown("### Prescription Recommendations")
            prescription = result.get('prescription', 'Prescriptions not available')
            st.markdown(prescription)
            
            # Download button for prescription
            st.download_button(
                label="📥 Download Prescriptions (TXT)",
                data=prescription,
                file_name="prescriptions.txt",
                mime="text/plain"
            )
        
        with report_tab3:
            st.markdown("### Health Suggestions & Lifestyle Recommendations")
            
            prediction = result['prediction']
            confidence = result['confidence']
            patient_info = st.session_state.patient_info
            
            # Generate suggestions based on risk factors
            suggestions = []
            
            # Risk factor analysis
            if int(patient_info['age']) > 50:
                suggestions.append("**Age Factor**: As you're over 50, regular health checkups are crucial.")
            
            if patient_info['smoking'] == 'Yes':
                suggestions.append("**Smoking**: Consider seeking professional help to quit smoking. This is one of the most modifiable risk factors.")
            
            if patient_info['high_blood_pressure'] == 'Yes':
                suggestions.append("**Blood Pressure**: Monitor your blood pressure regularly and follow medical advice for management.")
            
            if patient_info['diabetes'] == 'Yes':
                suggestions.append("**Diabetes**: Maintain strict glycemic control and regular diabetes monitoring.")
            
            if patient_info['exercise_habits'] == 'Low':
                suggestions.append("**Exercise**: Aim for at least 150 minutes of moderate-intensity aerobic activity per week.")
            
            if patient_info['stress_level'] == 'High':
                suggestions.append("**Stress Management**: Practice stress-reduction techniques like meditation, yoga, or deep breathing.")
            
            if patient_info['sleep_hours'] < '6':
                suggestions.append("**Sleep**: Aim for 7-9 hours of quality sleep per night.")
            
            if patient_info['alcohol_consumption'] == 'High':
                suggestions.append("**Alcohol**: Limit alcohol consumption to moderate levels (1 drink/day for women, 2 for men).")
            
            if patient_info['sugar_consumption'] == 'High':
                suggestions.append("**Diet**: Reduce refined sugars and processed foods. Focus on whole grains, fruits, and vegetables.")
            
            if patient_info['cholesterol'] > '240':
                suggestions.append("**Cholesterol**: Your cholesterol level is elevated. Consider dietary changes and/or medication.")
            
            if patient_info['bmi'] > '30':
                suggestions.append("**Weight Management**: Consider a weight management program. Aim for a BMI < 25.")
            
            if patient_info['family_heart_disease'] == 'Yes':
                suggestions.append("**Family History**: With a family history of heart disease, regular screening is important.")
            
            if suggestions:
                for i, suggestion in enumerate(suggestions, 1):
                    st.info(f"{i}. {suggestion}")
            else:
                st.success("✅ No major risk factors detected. Continue with healthy lifestyle practices.")
            
            # General recommendations
            st.markdown("---")
            st.markdown("### General Health Recommendations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Dietary Guidelines:**
                - Mediterranean or DASH diet
                - Low sodium intake
                - High in omega-3 fatty acids
                - Plenty of fruits and vegetables
                - Whole grains instead of refined carbs
                - Lean proteins
                """)
            
            with col2:
                st.markdown("""
                **Lifestyle Tips:**
                - Regular physical activity (cardio + strength)
                - Stress management techniques
                - Quality sleep (7-9 hours)
                - Maintain healthy weight
                - Avoid tobacco and limit alcohol
                - Regular health checkups
                """)
            
            # Follow-up recommendations
            st.markdown("---")
            st.markdown("### Follow-up Recommendations")
            
            if prediction == 1:
                st.warning("""
                **For High-Risk Patients:**
                - Schedule an appointment with a cardiologist
                - Get a complete cardiac evaluation
                - Consider stress testing or ECG
                - Monitor blood pressure daily
                - Keep all medical appointments
                """)
            else:
                st.success("""
                **For Low-Risk Patients:**
                - Annual health checkups
                - Maintain current healthy habits
                - Continue regular exercise
                - Monitor blood pressure periodically
                """)
    
    else:
        st.info("📝 Please generate a prediction first to see reports and recommendations.")

# ============================================================================
# EXPORT RESULTS
# ============================================================================
st.markdown("---")

if st.session_state.get('prediction_result', None):
    st.subheader("📊 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export as JSON
        import json
        result = st.session_state.prediction_result
        result_copy = result.copy()
        result_copy['confidence'] = float(result_copy['confidence'])
        result_copy['prediction'] = int(result_copy['prediction'])
        
        json_data = json.dumps(result_copy, indent=2)
        st.download_button(
            label="📥 Export Results (JSON)",
            data=json_data,
            file_name="prediction_results.json",
            mime="application/json"
        )
    
    with col2:
        # Export as CSV (Summary)
        summary_df = pd.DataFrame([st.session_state.patient_info])
        csv_data = summary_df.to_csv(index=False)
        st.download_button(
            label="📥 Export Patient Data (CSV)",
            data=csv_data,
            file_name="patient_data.csv",
            mime="text/csv"
        )
    
    with col3:
        st.info("💾 Use the download buttons to save your results")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #666;">
    <small>
    ❤️ Heart Disease Detection System | Educational Purpose Only<br>
    <strong>DISCLAIMER:</strong> This system is for educational and informational purposes only. 
    It is NOT a medical diagnosis. Always consult with qualified healthcare professionals.<br>
    </small>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'run_prediction' not in st.session_state:
    st.session_state.run_prediction = False
if 'patient_info' not in st.session_state:
    st.session_state.patient_info = {}
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
