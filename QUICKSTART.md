# 🚀 Quick Start Guide - Heart Disease Prediction GUI

## What You Have

A complete machine learning system with:
- ✅ Trained ML model for heart disease prediction
- ✅ LLM integration for medical reports and prescriptions
- ✅ Interactive Streamlit web GUI
- ✅ Export capabilities (JSON, CSV, TXT)

## Getting Started (3 Steps)

### Step 1: Export the Model (If not already done)

First, ensure the trained model is exported from the notebook.

**In `dataset.ipynb`, create a new cell at the end and run:**

```python
import joblib
import json

# Save the final classifier
joblib.dump(final_classifier, 'heart_disease_model.pkl')
print("✓ Model saved")

# Save the scaler
joblib.dump(scaler, 'heart_disease_scaler.pkl')
print("✓ Scaler saved")

# Save feature names
feature_names = X_train.columns.tolist()
with open('feature_names.json', 'w') as f:
    json.dump(feature_names, f)
print("✓ Features saved")

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
print("✓ Metadata saved")

print("\n✅ All model files exported successfully!")
```

### Step 2: Install Dependencies

```bash
# Navigate to project directory
cd "c:\Users\Track Computers\Desktop\HeartDisease Detector"

# Install required packages
pip install -r requirements.txt
```

**Or manually:**

```bash
pip install streamlit pandas numpy scikit-learn joblib python-dotenv openai
```

### Step 3: Configure API Key

Create a `.env` file in your project directory with:

```
OPENROUTER_API_KEY=your_api_key_here
```

## Run the Application

```bash
streamlit run app.py
```

The GUI will open automatically in your browser at `http://localhost:8501`

## System Architecture

```
┌─────────────────────────────────────────────────┐
│         Streamlit Web Interface (app.py)        │
│  ┌─────────────────────────────────────────┐    │
│  │ Tab 1: Patient Information Input        │    │
│  │ - Demographics                          │    │
│  │ - Vital Signs                           │    │
│  │ - Lab Tests                             │    │
│  │ - Lifestyle Factors                     │    │
│  │ - Medical History                       │    │
│  └─────────────────────────────────────────┘    │
│           ⬇️ Generate Prediction                │
│  ┌─────────────────────────────────────────┐    │
│  │ prediction.py (Core Logic)              │    │
│  │ ✓ Load ML Model                         │    │
│  │ ✓ Prepare Data                          │    │
│  │ ✓ Make Prediction                       │    │
│  │ ✓ Call LLM for Reports                  │    │
│  └─────────────────────────────────────────┘    │
│           ⬇️ Display Results                     │
│  ┌─────────────────────────────────────────┐    │
│  │ Tab 2: Prediction Results               │    │
│  │ - ML Prediction & Confidence            │    │
│  │ - Risk Assessment                       │    │
│  └─────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────┐    │
│  │ Tab 3: Reports & Recommendations        │    │
│  │ - Medical Report (from LLM)             │    │
│  │ - Prescriptions (from LLM)              │    │
│  │ - Health Suggestions                    │    │
│  └─────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

## Features Overview

### 📋 Patient Information Tab
- Easy-to-use form with 20+ health parameters
- Organized into logical sections
- Dropdown menus for categorical values
- Number inputs for numerical values
- Generate button to start analysis

### 🔍 Prediction Results Tab
- ML model prediction (disease detected/not detected)
- Confidence score (0-100%)
- Risk level classification
- Patient health summary table

### 📊 Reports & Recommendations Tab

**Three sub-tabs:**

1. **Medical Report**
   - Detailed analysis from AI assistant
   - Risk factor assessment
   - Complications analysis
   - Download as TXT file

2. **Prescriptions**
   - Medication recommendations
   - Lifestyle modifications
   - Monitoring guidelines
   - Follow-up schedules
   - Download as TXT file

3. **Suggestions**
   - Personalized health tips
   - Risk factor-specific advice
   - Dietary guidelines
   - Exercise recommendations
   - General health tips

## Example Workflow

1. **Fill Patient Information**
   - Enter patient age, gender, vitals
   - Input lab test results
   - Select lifestyle factors
   - Check medical history

2. **Generate Prediction**
   - Click "Generate Prediction & Report"
   - Wait for ML model to predict
   - Wait for LLM to generate reports

3. **Review Results**
   - Check prediction in Tab 2
   - Read detailed report in Tab 3
   - Review prescriptions and suggestions
   - Download results for records

4. **Export Results**
   - Medical Report (TXT)
   - Prescriptions (TXT)
   - Prediction Results (JSON)
   - Patient Data (CSV)

## File Structure

```
HeartDisease Detector/
├── app.py                           ← Streamlit GUI (NEW)
├── prediction.py                    ← Core prediction logic
├── llm.py                           ← LLM integration
├── dataset.ipynb                    ← Model training
├── heart_disease.csv                ← Training data
│
├── Model Files (after export)
├── heart_disease_model.pkl          ← Trained ML model
├── heart_disease_scaler.pkl         ← Data scaler
├── feature_names.json               ← Feature list
├── model_metadata.json              ← Model performance
│
├── Documentation
├── README_PREDICTION.md             ← Prediction system guide
├── STREAMLIT_SETUP.md              ← Streamlit setup guide
├── ARCHITECTURE.md                  ← System architecture
├── QUICKSTART.md                    ← This file
├── requirements.txt                 ← Python dependencies
│
└── .env                            ← API key (create this)
```

## Input Parameters Explained

| Parameter | Type | Range | Notes |
|-----------|------|-------|-------|
| Age | Number | 18-100 | In years |
| Gender | Dropdown | M/F | |
| Blood Pressure | Number | 80-250 | Systolic, mmHg |
| Cholesterol | Number | 100-400 | mg/dL |
| BMI | Number | 10-60 | Body Mass Index |
| Sleep Hours | Number | 0-12 | Per night |
| Triglycerides | Number | 0-500 | mg/dL |
| Fasting Sugar | Dropdown | Yes/No | > 120 mg/dL |
| CRP Level | Number | 0-20 | mg/L |
| Homocysteine | Number | 0-50 | μmol/L |
| Smoking | Dropdown | Yes/No | |
| Exercise | Dropdown | Low/Med/High | |
| Family History | Dropdown | Yes/No | Heart disease |
| Diabetes | Dropdown | Yes/No | |
| Hypertension | Dropdown | Yes/No | High BP |
| Low HDL | Dropdown | Yes/No | Bad cholesterol |
| High LDL | Dropdown | Yes/No | Good cholesterol |
| Stress Level | Dropdown | Low/Med/High | |
| Alcohol | Dropdown | Low/Med/High | Consumption |
| Sugar | Dropdown | Low/Med/High | Consumption |

## Prediction Output

The system returns:

```json
{
  "prediction": 1,
  "prediction_text": "Heart Disease Detected",
  "confidence": 0.85,
  "medical_report": "...(detailed AI-generated report)...",
  "prescription": "...(detailed prescriptions)...",
  "patient_info": {...}
}
```

## Performance Tips

1. **First Load**: May take 5-10 seconds to load model
2. **First Prediction**: May take 15-30 seconds (LLM API calls)
3. **Subsequent Predictions**: Much faster (cached results)
4. **Browser**: Use Chrome/Firefox for best experience
5. **Connection**: Ensure stable internet for LLM calls

## Troubleshooting

### Can't find model files
```
❌ "heart_disease_model.pkl not found"
✓ Run the export code from dataset.ipynb first
```

### API key error
```
❌ "OPENROUTER_API_KEY not found"
✓ Create .env file with your API key
```

### Port already in use
```bash
# Use different port
streamlit run app.py --server.port 8502
```

### Slow responses
```
❌ LLM taking too long
✓ Check internet connection
✓ Verify API key is valid
✓ Try again - first call is slower
```

## Important Notes

⚠️ **DISCLAIMER**: 
- This is an educational tool, NOT a medical diagnosis
- Always consult healthcare professionals
- Results are predictions, not diagnoses
- Use for awareness and information only

## Next Steps

1. ✅ Export model from dataset.ipynb
2. ✅ Install requirements: `pip install -r requirements.txt`
3. ✅ Create .env file with API key
4. ✅ Run app: `streamlit run app.py`
5. ✅ Fill patient info and generate predictions

## Need Help?

- **Model Issues**: See `README_PREDICTION.md`
- **Architecture**: See `ARCHITECTURE.md`
- **Streamlit Setup**: See `STREAMLIT_SETUP.md`
- **Streamlit Docs**: https://docs.streamlit.io

---

**Ready to use!** 🚀 Run `streamlit run app.py` to get started!
