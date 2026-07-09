"""
Heart Disease Prediction System
Uses trained ML model from dataset.ipynb for predictions
Uses LLM for medical reports, prescriptions, and suggestions
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

load_dotenv()

# Initialize OpenAI client for LLM
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Load trained model components
MODEL_DIR = Path(__file__).parent

def load_model_components():
    """Load the trained model, scaler, and feature names"""
    try:
        model = joblib.load(MODEL_DIR / 'heart_disease_model.pkl')
        scaler = joblib.load(MODEL_DIR / 'heart_disease_scaler.pkl')
        
        with open(MODEL_DIR / 'feature_names.json', 'r') as f:
            feature_names = json.load(f)
        
        with open(MODEL_DIR / 'model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        return model, scaler, feature_names, metadata
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Model files not found. Please ensure the following files exist:\n"
            f"  - heart_disease_model.pkl\n"
            f"  - heart_disease_scaler.pkl\n"
            f"  - feature_names.json\n"
            f"  - model_metadata.json\n\n"
            f"Run the export code in dataset.ipynb to generate these files.\n"
            f"Error: {e}"
        )

# Load model components at startup
try:
    FINAL_MODEL, SCALER, FEATURE_NAMES, MODEL_METADATA = load_model_components()
    MODEL_LOADED = True
except FileNotFoundError as e:
    print(f"⚠️  Warning: {e}")
    FINAL_MODEL = None
    SCALER = None
    FEATURE_NAMES = []
    MODEL_METADATA = {}
    MODEL_LOADED = False

def prepare_patient_data(patient_info):
    """
    Convert raw patient information to model input format matching notebook preprocessing
    
    Args:
        patient_info (dict): Raw patient information with categorical and numerical values
        
    Returns:
        np.array: Formatted features for model prediction
    """
    
    if not MODEL_LOADED:
        raise RuntimeError("Model not loaded. Cannot prepare data.")
    
    # Mapping dictionaries matching notebook encoding
    yes_no_map = {'Yes': 1, 'No': 0}
    gender_map = {'Male': 1, 'Female': 0}
    level_map = {'Low': 0, 'Medium': 1, 'High': 2}
    
    try:
        features = {}
        
        # Extract and encode patient data to match FEATURE_NAMES order
        features['Age'] = int(patient_info.get('age', 0))
        features['Gender'] = gender_map.get(patient_info.get('gender', 'Male'), 1)
        features['Blood Pressure'] = int(patient_info.get('blood_pressure', '120/80').split('/')[0])
        features['Cholesterol Level'] = int(patient_info.get('cholesterol', '200').replace(' mg/dL', ''))
        features['BMI'] = float(patient_info.get('bmi', '25.0'))
        features['Sleep Hours'] = float(patient_info.get('sleep_hours', '7.0'))
        features['Triglyceride Level'] = int(patient_info.get('triglyceride_level', '100'))
        features['Fasting Blood Sugar'] = yes_no_map.get(patient_info.get('fasting_blood_sugar', 'No'), 0)
        features['CRP Level'] = float(patient_info.get('crp_level', '3.0'))
        features['Homocysteine Level'] = float(patient_info.get('homocysteine_level', '10.0'))
        features['Smoking'] = yes_no_map.get(patient_info.get('smoking', 'No'), 0)
        features['Family Heart Disease'] = yes_no_map.get(patient_info.get('family_heart_disease', 'No'), 0)
        features['Diabetes'] = yes_no_map.get(patient_info.get('diabetes', 'No'), 0)
        features['High Blood Pressure'] = yes_no_map.get(patient_info.get('high_blood_pressure', 'No'), 0)
        features['Low HDL Cholesterol'] = yes_no_map.get(patient_info.get('low_hdl_cholesterol', 'No'), 0)
        features['High LDL Cholesterol'] = yes_no_map.get(patient_info.get('high_ldl_cholesterol', 'No'), 0)
        features['Exercise Habits'] = level_map.get(patient_info.get('exercise_habits', 'Medium'), 1)
        features['Stress Level'] = level_map.get(patient_info.get('stress_level', 'Medium'), 1)
        features['Alcohol Consumption'] = level_map.get(patient_info.get('alcohol_consumption', 'Low'), 0)
        features['Sugar Consumption'] = level_map.get(patient_info.get('sugar_consumption', 'Medium'), 1)
        
        # Create DataFrame with feature names in correct order
        data = pd.DataFrame([features])
        
        # Reorder columns to match FEATURE_NAMES
        data = data[FEATURE_NAMES]
        
        # Apply scaling to numerical features
        numerical_cols = [
            'Age', 'Blood Pressure', 'Cholesterol Level', 'BMI', 
            'Sleep Hours', 'Triglyceride Level', 'Fasting Blood Sugar', 
            'CRP Level', 'Homocysteine Level'
        ]
        
        data_scaled = data.copy()
        data_scaled[numerical_cols] = SCALER.transform(data[numerical_cols])
        
        return data_scaled.values
    
    except Exception as e:
        print(f"Error preparing patient data: {e}")
        raise


def generate_medical_report(patient_info, prediction, confidence):
    """
    Generate a medical report using LLM based on patient data and prediction
    
    Args:
        patient_info (dict): Patient information
        prediction (int): Model prediction (1 for disease, 0 for no disease)
        confidence (float): Prediction confidence (0-1)
        
    Returns:
        str: Medical report from LLM
    """
    
    prediction_text = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
    confidence_pct = f"{confidence * 100:.1f}%"
    
    prompt = f"""
You are a professional medical assistant. Analyze the following patient data and prediction from our heart disease detection model.

PATIENT INFORMATION:
- Age: {patient_info.get('age', 'N/A')} years
- Gender: {patient_info.get('gender', 'N/A')}
- Blood Pressure: {patient_info.get('blood_pressure', 'N/A')} mmHg
- Cholesterol Level: {patient_info.get('cholesterol', 'N/A')} mg/dL
- BMI: {patient_info.get('bmi', 'N/A')}
- Sleep Hours: {patient_info.get('sleep_hours', 'N/A')} hours
- Triglyceride Level: {patient_info.get('triglyceride_level', 'N/A')}
- Fasting Blood Sugar: {patient_info.get('fasting_blood_sugar', 'N/A')}
- CRP Level: {patient_info.get('crp_level', 'N/A')}
- Homocysteine Level: {patient_info.get('homocysteine_level', 'N/A')}
- Smoking: {patient_info.get('smoking', 'N/A')}
- Family History of Heart Disease: {patient_info.get('family_heart_disease', 'N/A')}
- Diabetes: {patient_info.get('diabetes', 'N/A')}
- High Blood Pressure: {patient_info.get('high_blood_pressure', 'N/A')}
- Low HDL Cholesterol: {patient_info.get('low_hdl_cholesterol', 'N/A')}
- High LDL Cholesterol: {patient_info.get('high_ldl_cholesterol', 'N/A')}
- Exercise Habits: {patient_info.get('exercise_habits', 'N/A')}
- Stress Level: {patient_info.get('stress_level', 'N/A')}
- Alcohol Consumption: {patient_info.get('alcohol_consumption', 'N/A')}
- Sugar Consumption: {patient_info.get('sugar_consumption', 'N/A')}

MODEL PREDICTION: {prediction_text}
PREDICTION CONFIDENCE: {confidence_pct}

Please provide a comprehensive medical report including:
1. Assessment of the prediction and confidence level
2. Identified risk factors based on the patient data
3. Possible complications if disease is present
4. Key health metrics to monitor
5. IMPORTANT: Remind that this is NOT a medical diagnosis and professional consultation is essential

Keep the response professional and clear for both medical and non-medical readers."""

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-3.1-8b-instruct",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating report: {str(e)}"


def generate_prescription(patient_info, prediction):
    """
    Generate prescription recommendations using LLM
    
    Args:
        patient_info (dict): Patient information
        prediction (int): Model prediction (1 for disease, 0 for no disease)
        
    Returns:
        str: Prescription recommendations from LLM
    """
    
    prediction_text = "Heart Disease Risk Detected" if prediction == 1 else "No Heart Disease Risk"
    
    prompt = f"""
You are a medical professional. Based on the patient's profile and the disease prediction, provide comprehensive treatment recommendations.

PATIENT PROFILE:
- Age: {patient_info.get('age', 'N/A')}
- Gender: {patient_info.get('gender', 'N/A')}
- Blood Pressure: {patient_info.get('blood_pressure', 'N/A')}
- Cholesterol: {patient_info.get('cholesterol', 'N/A')}
- Diabetes: {patient_info.get('diabetes', 'N/A')}
- Smoking: {patient_info.get('smoking', 'N/A')}
- Exercise Habits: {patient_info.get('exercise_habits', 'N/A')}
- Stress Level: {patient_info.get('stress_level', 'N/A')}

DIAGNOSIS STATUS: {prediction_text}

Please provide:

1. MEDICATION RECOMMENDATIONS (if disease risk is present):
   - List potential medications by category (blood pressure, cholesterol, etc.)
   - Note: These are suggestions only - actual medications must be prescribed by a doctor

2. LIFESTYLE MODIFICATIONS:
   - Diet recommendations
   - Exercise program suggestions
   - Stress management techniques
   - Smoking cessation (if applicable)

3. MONITORING & FOLLOW-UP:
   - Recommended tests/checkups frequency
   - When to seek emergency care
   - Red flag symptoms to watch for

4. PREVENTION STRATEGIES:
   - For high-risk patients: aggressive prevention measures
   - For low-risk patients: maintenance recommendations

DISCLAIMER: These are general recommendations only. Always consult with a qualified healthcare provider for personalized medical advice and prescriptions."""

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-3.1-8b-instruct",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating prescription: {str(e)}"


def predict_heart_disease(patient_info):
    """
    Main function to predict heart disease and generate reports
    
    Args:
        patient_info (dict): Patient information (should match expected features)
        
    Returns:
        dict: Prediction result, report, and prescriptions
    """
    
    if not MODEL_LOADED:
        print("✗ Error: Model not loaded. Please ensure model files are available.")
        return None
    
    print("=" * 80)
    print("HEART DISEASE PREDICTION SYSTEM")
    print(f"Model: {MODEL_METADATA.get('model_name', 'Unknown')}")
    print("=" * 80)
    
    # Prepare data
    print("\n[1/4] Preparing patient data...")
    try:
        X = prepare_patient_data(patient_info)
        print("✓ Patient data prepared successfully")
    except Exception as e:
        print(f"✗ Error preparing data: {e}")
        return None
    
    # Make prediction
    print("\n[2/4] Generating prediction...")
    try:
        prediction = FINAL_MODEL.predict(X)[0]
        confidence = FINAL_MODEL.predict_proba(X)[0].max()
        
        print(f"✓ Prediction: {'Heart Disease Detected' if prediction == 1 else 'No Heart Disease'}")
        print(f"✓ Confidence: {confidence:.2%}")
    except Exception as e:
        print(f"✗ Error making prediction: {e}")
        return None
    
    # Generate medical report
    print("\n[3/4] Generating medical report...")
    try:
        report = generate_medical_report(patient_info, prediction, confidence)
        print("✓ Medical report generated")
    except Exception as e:
        print(f"✗ Error generating report: {e}")
        report = "Report generation failed"
    
    # Generate prescription
    print("\n[4/4] Generating prescription recommendations...")
    try:
        prescription = generate_prescription(patient_info, prediction)
        print("✓ Prescription generated")
    except Exception as e:
        print(f"✗ Error generating prescription: {e}")
        prescription = "Prescription generation failed"
    
    # Prepare result
    result = {
        'patient_info': patient_info,
        'prediction': prediction,
        'prediction_text': 'Heart Disease Detected' if prediction == 1 else 'No Heart Disease',
        'confidence': confidence,
        'medical_report': report,
        'prescription': prescription
    }
    
    return result


def display_results(result):
    """
    Display prediction results in a formatted way
    
    Args:
        result (dict): Result dictionary from predict_heart_disease
    """
    
    if result is None:
        print("No results to display")
        return
    
    print("\n" + "=" * 80)
    print("PREDICTION RESULTS")
    print("=" * 80)
    
    print(f"\nPrediction: {result['prediction_text']}")
    print(f"Confidence: {result['confidence']:.2%}")
    
    print("\n" + "-" * 80)
    print("MEDICAL REPORT")
    print("-" * 80)
    print(result['medical_report'])
    
    print("\n" + "-" * 80)
    print("PRESCRIPTION RECOMMENDATIONS")
    print("-" * 80)
    print(result['prescription'])
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Example usage - high risk patient
    sample_patient = {
        'age': '55',
        'gender': 'Male',
        'blood_pressure': '150',
        'cholesterol': '280',
        'bmi': '30.5',
        'sleep_hours': '6.0',
        'triglyceride_level': '200',
        'fasting_blood_sugar': 'Yes',
        'crp_level': '5.2',
        'homocysteine_level': '12.5',
        'smoking': 'Yes',
        'family_heart_disease': 'Yes',
        'diabetes': 'Yes',
        'high_blood_pressure': 'Yes',
        'low_hdl_cholesterol': 'Yes',
        'high_ldl_cholesterol': 'Yes',
        'exercise_habits': 'Low',
        'stress_level': 'High',
        'alcohol_consumption': 'Medium',
        'sugar_consumption': 'High'
    }
    
    # Run prediction
    result = predict_heart_disease(sample_patient)
    
    # Display results
    if result:
        display_results(result)
        
        # Optionally save results to JSON
        with open('prediction_result.json', 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            result_copy = result.copy()
            result_copy['confidence'] = float(result_copy['confidence'])
            result_copy['prediction'] = int(result_copy['prediction'])
            json.dump(result_copy, f, indent=2)
        print("\n✓ Results saved to 'prediction_result.json'")
