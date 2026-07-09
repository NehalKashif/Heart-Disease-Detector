# Updated Heart Disease Prediction System - Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Heart Disease Prediction System - Architecture               │
└─────────────────────────────────────────────────────────────────┘

INPUT: Patient Health Data
         ↓
    ┌────────────────────┐
    │  prediction.py     │ ← Main prediction module
    │  ─────────────────ー │
    │  1. Load trained   │
    │     ML model       │  (from dataset.ipynb)
    │  2. Prepare data   │
    │  3. Make          │
    │     prediction    │
    └────────────────────┘
              ↓
    ┌────────────────────────────────────────┐
    │  ML Prediction Result:                 │
    │  - Prediction (0 or 1)                 │
    │  - Confidence Score (%)                │
    └────────────────────────────────────────┘
              ↓ (Feed to LLM)
    ┌────────────────────┐
    │   LLM (llm.py)     │
    │  ──────────────────ー │
    │  Uses OpenRouter   │
    └────────────────────┘
         ↙      ↓      ↘
    ┌──────┐ ┌──────┐ ┌──────────┐
    │Report│ │Rx    │ │Suggest   │
    │      │ │      │ │          │
    └──────┘ └──────┘ └──────────┘
         ↓
    OUTPUT: Complete Patient Analysis
```

## File Structure

```
HeartDisease Detector/
├── dataset.ipynb              # ML model training
├── heart_disease.csv          # Training data
├── prediction.py              # ★ NEW - Main prediction system
├── llm.py                      # LLM integration
├── README_PREDICTION.md        # ★ NEW - Setup instructions
├── export_model.py             # ★ NEW - Model export guide
│
├── [Generated Files - After model export]
├── heart_disease_model.pkl     # Trained ML model
├── heart_disease_scaler.pkl    # Data preprocessor
├── feature_names.json          # Feature column names
└── model_metadata.json         # Model performance metrics
```

## Workflow

### Phase 1: Training (In dataset.ipynb)
```
1. Load heart_disease.csv
2. Clean & preprocess data
3. Train 6 different ML models
4. Select best model (by recall)
5. Export model files using new cell:
   - heart_disease_model.pkl
   - heart_disease_scaler.pkl
   - feature_names.json
   - model_metadata.json
```

### Phase 2: Prediction (Using prediction.py)
```
1. Load saved model & scaler
2. Accept patient health data
3. Prepare data (encode + scale)
4. Make prediction using ML model
5. Get confidence score
6. Pass results to LLM
7. LLM generates:
   - Medical report
   - Prescriptions
   - Recommendations
8. Display & save results
```

## Key Changes from Original

| Aspect | Before | After |
|--------|--------|-------|
| Model | Optional parameter | Loaded from pickle file |
| Data Prep | Manual encoding | Automated with scaler |
| Features | Generic list | Loaded from JSON |
| LLM Usage | Optional | Always used for reports |
| Patient Data | Custom format | Notebook-aligned format |
| Scaler | Not used | Applied to numerical features |

## Feature Mapping

The system now expects exactly these fields (matching notebook):

```
{'age': '55',                          # Numerical
 'gender': 'Male',                     # Male/Female
 'blood_pressure': '150',              # Numerical
 'cholesterol': '280',                 # Numerical
 'bmi': '30.5',                        # Numerical (Body Mass Index)
 'sleep_hours': '6.0',                 # Numerical
 'triglyceride_level': '200',          # Numerical
 'fasting_blood_sugar': 'Yes',         # Yes/No
 'crp_level': '5.2',                   # Numerical (C-Reactive Protein)
 'homocysteine_level': '12.5',         # Numerical
 'smoking': 'Yes',                     # Yes/No
 'family_heart_disease': 'Yes',        # Yes/No
 'diabetes': 'Yes',                    # Yes/No
 'high_blood_pressure': 'Yes',         # Yes/No
 'low_hdl_cholesterol': 'Yes',         # Yes/No
 'high_ldl_cholesterol': 'Yes',        # Yes/No
 'exercise_habits': 'Low',             # Low/Medium/High
 'stress_level': 'High',               # Low/Medium/High
 'alcohol_consumption': 'Medium',      # Low/Medium/High
 'sugar_consumption': 'High'}          # Low/Medium/High
```

## Running the System

### Quick Start
```bash
# 1. Export model from notebook first!
# (Run the export cell in dataset.ipynb)

# 2. Run prediction
python prediction.py

# Output will include:
# - Model info
# - Prediction & confidence
# - Medical report (from LLM)
# - Prescriptions (from LLM)
# - Results saved to JSON
```

### Programmatic Usage
```python
from prediction import predict_heart_disease, display_results

patient = {'age': '55', 'gender': 'Male', ...}
result = predict_heart_disease(patient)
display_results(result)
```

## Important Notes

✅ **What's New:**
- Actual ML model predictions (not random)
- Automatic data scaling & preprocessing
- Feature validation
- Model metadata tracking

⚠️ **Requirements:**
- Must export model files from dataset.ipynb first
- Model, scaler, and feature names must be available
- OpenRouter API key in .env file

📋 **Disclaimer:**
NOT a medical diagnosis - for educational/informational use only
Always consult qualified healthcare providers

## Error Handling

If you get "Model files not found":
1. Run all cells in dataset.ipynb
2. Run the export cell at the end
3. Verify these files exist:
   - heart_disease_model.pkl
   - heart_disease_scaler.pkl
   - feature_names.json
   - model_metadata.json

If prediction fails:
1. Check patient data format
2. Verify enum values (Yes/No, Low/Medium/High)
3. Check console for specific error
4. See README_PREDICTION.md for troubleshooting
