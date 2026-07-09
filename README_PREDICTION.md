# Heart Disease Prediction System

## Overview
This system uses a machine learning model trained in `dataset.ipynb` to predict heart disease risk, combined with an LLM for generating medical reports, prescriptions, and recommendations.

## Setup Instructions

### Step 1: Train the Model (in dataset.ipynb)
1. Run all cells in `dataset.ipynb` to train the models
2. The notebook will select the best model based on recall score

### Step 2: Export the Model
Add this code to a **new cell** at the end of `dataset.ipynb`:

```python
import joblib
import json

# Save the final classifier
joblib.dump(final_classifier, 'heart_disease_model.pkl')
print("✓ Model saved: heart_disease_model.pkl")

# Save the scaler
joblib.dump(scaler, 'heart_disease_scaler.pkl')
print("✓ Scaler saved: heart_disease_scaler.pkl")

# Save feature names
feature_names = X_train.columns.tolist()
with open('feature_names.json', 'w') as f:
    json.dump(feature_names, f)
print("✓ Features saved: feature_names.json")

# Save model metadata
model_info = {
    'model_name': final_model_name,
    'accuracy': float(final_accuracy),
    'precision': float(final_precision),
    'recall': float(final_recall),
    'auc': float(final_auc),
    'feature_count': len(feature_names)
}
with open('model_metadata.json', 'w') as f:
    json.dump(model_info, f, indent=2)
print("✓ Metadata saved: model_metadata.json")
print("\n✅ All model files exported successfully!")
```

This will create 4 files in the workspace:
- `heart_disease_model.pkl` - The trained ML model
- `heart_disease_scaler.pkl` - The data scaler
- `feature_names.json` - Feature column names
- `model_metadata.json` - Model performance metrics

### Step 3: Set Up Environment
1. Create a `.env` file with your OpenRouter API key:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```

2. Install required packages:
   ```bash
   pip install joblib pandas scikit-learn openai python-dotenv
   ```

## Usage

### Run Predictions
```bash
python prediction.py
```

This will:
1. Load the trained model
2. Prepare patient data
3. Make a prediction with confidence score
4. Generate a medical report via LLM
5. Generate prescription recommendations via LLM
6. Display all results
7. Save results to `prediction_result.json`

### Use in Your Code
```python
from prediction import predict_heart_disease, display_results

patient_info = {
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

result = predict_heart_disease(patient_info)
if result:
    display_results(result)
```

## Patient Information Fields

| Field | Type | Example |
|-------|------|---------|
| age | string | '55' |
| gender | string | 'Male' or 'Female' |
| blood_pressure | string | '150' |
| cholesterol | string | '280' |
| bmi | string | '30.5' |
| sleep_hours | string | '6.0' |
| triglyceride_level | string | '200' |
| fasting_blood_sugar | string | 'Yes' or 'No' |
| crp_level | string | '5.2' |
| homocysteine_level | string | '12.5' |
| smoking | string | 'Yes' or 'No' |
| family_heart_disease | string | 'Yes' or 'No' |
| diabetes | string | 'Yes' or 'No' |
| high_blood_pressure | string | 'Yes' or 'No' |
| low_hdl_cholesterol | string | 'Yes' or 'No' |
| high_ldl_cholesterol | string | 'Yes' or 'No' |
| exercise_habits | string | 'Low', 'Medium', or 'High' |
| stress_level | string | 'Low', 'Medium', or 'High' |
| alcohol_consumption | string | 'Low', 'Medium', or 'High' |
| sugar_consumption | string | 'Low', 'Medium', or 'High' |

## Output

The system returns:
- **Prediction**: Heart Disease Detected or No Heart Disease
- **Confidence**: Probability score (0-100%)
- **Medical Report**: Detailed analysis from LLM
- **Prescription**: Treatment recommendations from LLM
- **Results JSON**: Saved to `prediction_result.json`

## Important Notes

⚠️ **DISCLAIMER**: This system is for educational and informational purposes only. It is NOT a medical diagnosis. Always consult with a qualified healthcare provider for actual medical advice.

## Troubleshooting

### "Model files not found"
- Ensure you ran the export code in `dataset.ipynb`
- Check that these files exist in the workspace:
  - `heart_disease_model.pkl`
  - `heart_disease_scaler.pkl`
  - `feature_names.json`
  - `model_metadata.json`

### "OPENROUTER_API_KEY not found"
- Create a `.env` file with your API key
- Format: `OPENROUTER_API_KEY=your_key_here`

### Prediction errors
- Verify all patient information fields are provided
- Check that enum values match allowed options (Yes/No, Low/Medium/High)
