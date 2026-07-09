# Heart Disease Prediction GUI - Streamlit Setup Guide

## Overview

The Streamlit GUI provides an interactive web interface for the Heart Disease Prediction System. It allows users to input patient health data and get predictions, medical reports, and prescriptions.

## Prerequisites

Before running the Streamlit app, ensure you have:

1. **Trained Model Files** (from `dataset.ipynb`):
   - `heart_disease_model.pkl`
   - `heart_disease_scaler.pkl`
   - `feature_names.json`
   - `model_metadata.json`

2. **Environment Variables** (`.env` file):
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```

## Installation

### Step 1: Install Requirements

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install streamlit pandas numpy scikit-learn joblib python-dotenv openai xgboost
```

### Step 2: Verify Model Files

Make sure these files exist in the project directory:
```
HeartDisease Detector/
├── heart_disease_model.pkl
├── heart_disease_scaler.pkl
├── feature_names.json
└── model_metadata.json
```

If they don't exist, run the export code in `dataset.ipynb`:

```python
import joblib
import json

# Save model components
joblib.dump(final_classifier, 'heart_disease_model.pkl')
joblib.dump(scaler, 'heart_disease_scaler.pkl')

feature_names = X_train.columns.tolist()
with open('feature_names.json', 'w') as f:
    json.dump(feature_names, f)

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
```

### Step 3: Create `.env` File

In the project root directory, create a `.env` file with your OpenRouter API key:

```
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxx
```

## Running the Application

### Option 1: Using Command Line

```bash
streamlit run app.py
```

The app will automatically open in your default browser at `http://localhost:8501`

### Option 2: Specify Port (if default is in use)

```bash
streamlit run app.py --server.port 8502
```

### Option 3: Using Python

```bash
python -m streamlit run app.py
```

## Using the Application

### Tab 1: Patient Information Input

1. **Fill in Patient Demographics:**
   - Age (18-100 years)
   - Gender (Male/Female)

2. **Enter Vital Signs:**
   - Blood Pressure (Systolic, mmHg)
   - Cholesterol Level (mg/dL)

3. **Body Metrics:**
   - BMI (Body Mass Index)
   - Sleep Hours (per night)

4. **Laboratory Tests:**
   - Triglyceride Level
   - CRP Level (C-Reactive Protein)
   - Fasting Blood Sugar (Yes/No)
   - Homocysteine Level

5. **Lifestyle Factors:**
   - Smoking (Yes/No)
   - Exercise Habits (Low/Medium/High)

6. **Medical History:**
   - Family History of Heart Disease
   - Diabetes (Yes/No)
   - High Blood Pressure (Yes/No)
   - Low/High HDL/LDL Cholesterol

7. **Stress & Diet:**
   - Stress Level (Low/Medium/High)
   - Alcohol Consumption (Low/Medium/High)
   - Sugar Consumption (Low/Medium/High)

8. **Click "Generate Prediction & Report"** to analyze the patient data

### Tab 2: Prediction Results

- **Prediction Status**: Whether heart disease is detected
- **Confidence Score**: Probability of the prediction
- **Risk Level**: High/Moderate/Low risk classification
- **Patient Summary**: Quick overview of key metrics

### Tab 3: Reports & Recommendations

Three sub-tabs with comprehensive information:

1. **Medical Report**: Detailed analysis of the prediction and risk factors
2. **Prescriptions**: Medication suggestions and treatment recommendations
3. **Suggestions**: Personalized health recommendations and lifestyle tips

## Features

### Interactive Input
- Easy-to-use form with all patient health parameters
- Real-time form validation
- Clear sections for different health aspects

### Instant Predictions
- ML model predictions with confidence scores
- Color-coded risk indicators
- Quick patient health summary

### Comprehensive Reports
- Medical analysis from AI assistant
- Prescription recommendations
- Personalized health suggestions

### Export Options
- Download medical report (TXT)
- Download prescriptions (TXT)
- Export prediction results (JSON)
- Export patient data (CSV)

### Model Information
- Displays model type and performance metrics
- Shows accuracy, recall, and ROC-AUC scores
- Real-time model status

## Troubleshooting

### Issue: "Model files not found"

**Solution:**
1. Verify model files exist in the directory
2. Run the export code in `dataset.ipynb`
3. Check file names match exactly:
   - `heart_disease_model.pkl`
   - `heart_disease_scaler.pkl`
   - `feature_names.json`
   - `model_metadata.json`

### Issue: "OPENROUTER_API_KEY not found"

**Solution:**
1. Create `.env` file in project root
2. Add your API key: `OPENROUTER_API_KEY=your_key_here`
3. Restart the Streamlit app

### Issue: Port 8501 already in use

**Solution:**
Use a different port:
```bash
streamlit run app.py --server.port 8502
```

### Issue: Dependencies not installing

**Solution:**
```bash
# Update pip
python -m pip install --upgrade pip

# Install requirements with no cache
pip install --no-cache-dir -r requirements.txt
```

### Issue: Slow LLM responses

**Solution:**
- First response may take longer due to API initialization
- Subsequent responses should be faster
- Check internet connection
- Verify OpenRouter API key is valid

## Keyboard Shortcuts (Streamlit)

- `R` - Rerun the app
- `C` - Clear cache
- `P` - Print debug info

## Performance Tips

1. **First Run**: The LLM may take 10-30 seconds to generate responses
2. **Caching**: Streamlit caches results - repeated predictions are faster
3. **API Limits**: Be aware of OpenRouter API rate limits
4. **Browser**: Use a modern browser (Chrome, Firefox, Safari)

## Production Deployment

For deploying to production, consider:

1. **Streamlit Cloud** (free):
   ```bash
   git push # push to GitHub
   # Then deploy from Streamlit Cloud dashboard
   ```

2. **Docker**:
   ```dockerfile
   FROM python:3.10
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   CMD ["streamlit", "run", "app.py"]
   ```

3. **AWS/Azure/GCP**: Use container services

## Security Considerations

- Never commit `.env` file to git
- Use environment variables for API keys
- Consider HTTPS for production
- Limit API access with authentication
- Monitor API usage for unusual activity

## Support

For issues or questions:
1. Check the README_PREDICTION.md for model-specific info
2. Review ARCHITECTURE.md for system design
3. Check Streamlit documentation: https://docs.streamlit.io
4. Verify your OpenRouter API key is valid

## Disclaimer

⚠️ **IMPORTANT**: This system is for educational and informational purposes only. It is NOT a medical diagnosis. Always consult with qualified healthcare professionals for medical advice.

---

**Version**: 1.0  
**Last Updated**: 2024  
**Author**: Heart Disease Prediction Project
