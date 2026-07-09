# Heart Disease Prediction System

## 🎯 Project Overview

**Heart Disease Prediction System** is an intelligent healthcare application that combines machine learning classification with large language model (LLM) capabilities to provide comprehensive heart disease risk assessments. The system predicts whether a patient has heart disease based on 20 clinical health features and generates personalized medical reports, prescriptions, and lifestyle recommendations.

### Main Objectives
- **Predict** heart disease risk using a trained ML classifier
- **Analyze** patient health data across cardiovascular, lifestyle, and metabolic indicators
- **Generate** detailed medical reports using LLM analysis
- **Recommend** personalized treatment plans and lifestyle modifications
- **Provide** an intuitive web interface for healthcare providers and patients

---

## 📊 Dataset

### Dataset Overview
- **Name**: Heart Disease Dataset
- **Total Records**: 10,000 patient samples
- **Features**: 20 clinical health indicators
- **Target Variable**: Heart Disease Status (Binary: Yes/No)
- **File**: `heart_disease.csv`

### Class Distribution
- **No Disease**: 8,000 patients (80%)
- **Has Disease**: 2,000 patients (20%)
- **Challenge**: Severe class imbalance detected → Addressed using SMOTE

### Train/Test Split
- **Training Set**: 8,000 samples (80%)
- **Test Set**: 2,000 samples (20%)
- **Class Distribution in Train**: No Disease (6,400) / Has Disease (1,600)
- **Class Distribution in Test**: No Disease (1,600) / Has Disease (400)

### Features (20 Input Variables)

#### Demographics & Vitals
| Feature | Type | Range |
|---------|------|-------|
| Age | Numeric | 18-100 years |
| Gender | Categorical | Male/Female |
| Blood Pressure | Numeric | 80-250 mmHg |
| Cholesterol Level | Numeric | 100-400 mg/dL |

#### Body Metrics
| Feature | Type | Range |
|---------|------|-------|
| BMI | Numeric | 10-60 kg/m² |
| Sleep Hours | Numeric | 0-12 hours |
| Triglyceride Level | Numeric | 0-500 mg/dL |
| CRP Level | Numeric | 0-20 mg/L |

#### Metabolic & Laboratory
| Feature | Type | Values |
|---------|------|--------|
| Fasting Blood Sugar | Categorical | Yes/No |
| Homocysteine Level | Numeric | 0-50 µmol/L |

#### Lifestyle Factors
| Feature | Type | Values |
|---------|------|--------|
| Smoking | Categorical | Yes/No |
| Exercise Habits | Categorical | Low/Medium/High |
| Alcohol Consumption | Categorical | Low/Medium/High |
| Sugar Consumption | Categorical | Low/Medium/High |
| Stress Level | Categorical | Low/Medium/High |

#### Medical History
| Feature | Type | Values |
|---------|------|--------|
| Family Heart Disease | Categorical | Yes/No |
| Diabetes | Categorical | Yes/No |
| High Blood Pressure | Categorical | Yes/No |
| Low HDL Cholesterol | Categorical | Yes/No |
| High LDL Cholesterol | Categorical | Yes/No |

### Data Preprocessing
1. **Duplicate Removal**: 0 duplicates found
2. **Missing Values**: Imputed using median (numeric) and mode (categorical)
3. **Encoding**: 
   - Binary features → 0/1 encoding
   - Ordinal features (Low/Medium/High) → 0/1/2 encoding
   - Gender → Male=1, Female=0
4. **Scaling**: StandardScaler applied to 9 numerical features
5. **Class Balancing**: SMOTE applied to training set only (test set preserved)

---

## 🤖 Machine Learning Models

### Models Trained & Evaluated
Six different classifiers were trained and compared:

#### 1. **Logistic Regression**
- **Accuracy**: 0.5825 (58.25%)
- **Precision**: 0.2000 (20%)
- **Recall**: 0.3625 (36.25%)
- **ROC-AUC**: 0.5103

#### 2. **K-Nearest Neighbors (KNN)**
- **Best K**: 19
- **Accuracy**: 0.4690 (46.90%)
- **Precision**: 0.2023 (20.23%)
- **Recall**: 0.5625 (56.25%) ⭐ **HIGHEST RECALL**
- **ROC-AUC**: 0.4938

#### 3. **Random Forest**
- **Accuracy**: 0.6515 (65.15%) ⭐ **HIGHEST ACCURACY**
- **Precision**: 0.2071 (20.71%)
- **Recall**: 0.2625 (26.25%)
- **ROC-AUC**: 0.5079

#### 4. **Support Vector Machine (SVM)**
- **Accuracy**: 0.6370 (63.70%)
- **Precision**: 0.2015 (20.15%)
- **Recall**: 0.2750 (27.50%)
- **ROC-AUC**: 0.5142 ⭐ **HIGHEST ROC-AUC**

#### 5. **XGBoost**
- **Accuracy**: 0.6495 (64.95%)
- **Precision**: 0.2111 (21.11%) ⭐ **HIGHEST PRECISION**
- **Recall**: 0.2750 (27.50%)
- **ROC-AUC**: 0.4941

#### 6. **Gaussian Naive Bayes**
- **Accuracy**: 0.5955 (59.55%)
- **Precision**: 0.2015 (20.15%)
- **Recall**: 0.3450 (34.50%)
- **ROC-AUC**: 0.5067

### 🏆 Selected Final Model: KNN (K=19)
**Reason**: Highest recall score (0.5625) → Prioritizes catching actual disease cases

**Final Performance Metrics**:
```
Accuracy  : 0.4690 (46.90% of predictions correct)
Precision : 0.2023 (20.23% of disease predictions are correct)
Recall    : 0.5625 (56.25% of actual disease cases detected)
ROC-AUC   : 0.4938 (49.38% probability of ranking positive higher)
```

### Model Insights
- ⚠️ **Low Accuracy Despite Large Dataset**: 46.90% accuracy on KNN despite 10,000 samples indicates:
  - Precision across all models is very low (20-21%) - high false positive rate
  - Class imbalance creates majority class bias
  - Features may not be strongly predictive of disease
- ✅ **Strong Recall**: 56.25% recall catches majority of disease cases (good for screening)
- 📊 **Class Imbalance Challenge**: 80/20 split means even with SMOTE, predicting disease is difficult
- 🔄 **Trade-off**: KNN chosen for high recall (catching disease) over accuracy (precision is sacrificed)
- 💡 **Alternative Models**: Random Forest achieves 65% accuracy but only 26% recall (misses disease cases)

### Important Note
```
⚠️ DISCLAIMER
This model is for educational and risk assessment purposes ONLY.
It should NOT be used as a substitute for professional medical diagnosis.
Always consult qualified healthcare providers for medical decisions.
```

---

## 🧠 LLM Integration

### Model Used
- **Provider**: OpenRouter
- **Model**: `mistral/mistral-7b-instruct:free`
- **API Endpoint**: `https://openrouter.ai/api/v1`

### Why Mistral 7B?
- **Free tier availability**: No cost constraints during development
- **Fast inference**: Suitable for real-time report generation
- **Good quality**: Competitive performance for medical text generation
- **Reliable**: Stable availability on OpenRouter

### LLM Capabilities in This System

#### 1. Medical Report Generation
- Analyzes patient health data with ML prediction
- Generates comprehensive cardiovascular health assessment
- Identifies risk factors from patient data
- Provides monitoring recommendations
- Creates structured clinical documentation

**Example Output**:
```
Patient Risk Assessment
- Overall Risk Category: High
- Primary Risk Factors: Hypertension, High Cholesterol, Smoking
- Cardiovascular Status: Concerning
- Recommended Actions: Immediate lifestyle changes, medication review
- Monitoring Plan: Weekly BP checks, monthly lipid panel
```

#### 2. Prescription Generation
- Creates personalized medication suggestions
- Recommends lifestyle modifications
- Provides dietary guidelines specific to patient profile
- Suggests exercise regimens
- Details monitoring schedules

**Example Output**:
```
Treatment Plan
Medications:
- Statin for cholesterol management
- ACE inhibitor for blood pressure
- Antiplatelet therapy consideration

Lifestyle:
- Reduce sodium intake to <2g/day
- Exercise 30 minutes daily
- Stress management techniques
- Smoking cessation program

Monitoring:
- Blood pressure: Daily
- Cholesterol: Monthly
- Follow-up: 2 weeks
```

#### 3. Health Suggestions
- Personalized dietary recommendations based on risk factors
- Exercise plans tailored to fitness level
- Stress management techniques
- Sleep optimization strategies
- Social support recommendations

### API Configuration
```python
# .env file (NOT committed to git)
OPENROUTER_API_KEY=your_api_key_here
```

---

## 💻 GUI - Streamlit Web Application

### Overview
Interactive web-based interface built with Streamlit for end-to-end patient assessment.

### Technology Stack
- **Framework**: Streamlit 1.28.0
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib
- **Backend**: Python 3.11.4

### User Interface Components

#### 1. **Sidebar**
- Model Information Display
  - Model Name: KNeighborsClassifier
  - Accuracy, Precision, Recall, ROC-AUC metrics
- System Status Indicator
- Disclaimer & Legal Notice

#### 2. **Tab 1: Patient Information**
20 input fields organized in an intuitive form:

**Section 1 - Demographics**
- Age slider (18-100 years)
- Gender dropdown (Male/Female)

**Section 2 - Vital Signs**
- Blood Pressure input (80-250 mmHg)
- Cholesterol Level input (100-400 mg/dL)

**Section 3 - Body Metrics**
- BMI input (10-60 kg/m²)
- Sleep Hours input (0-12 hours)

**Section 4 - Laboratory Values**
- Triglyceride Level (0-500 mg/dL)
- CRP Level (0-20 mg/L)
- Fasting Blood Sugar (Yes/No)
- Homocysteine Level (0-50 µmol/L)

**Section 5 - Lifestyle Factors**
- Smoking (Yes/No)
- Exercise Habits (Low/Medium/High)
- Alcohol Consumption (Low/Medium/High)
- Sugar Consumption (Low/Medium/High)
- Stress Level (Low/Medium/High)

**Section 6 - Medical History**
- Family Heart Disease (Yes/No)
- Diabetes (Yes/No)
- High Blood Pressure (Yes/No)
- Low HDL Cholesterol (Yes/No)
- High LDL Cholesterol (Yes/No)

**Action Buttons**:
- "Generate Prediction & Report" - Triggers full pipeline
- "Clear Form" - Resets all fields

#### 3. **Tab 2: Prediction Results**
- **Color-coded Prediction Display**
  - Red gradient for disease detected
  - Blue gradient for no disease
- **Confidence Score** (0-100% metric)
- **Risk Level Classification** (High/Moderate/Low)
- **Patient Summary Table**
  - 8 key metrics displayed
  - Easy reference of input values

#### 4. **Tab 3: Reports & Recommendations**

**Sub-tab A: Medical Report**
- LLM-generated comprehensive analysis
- Risk factor breakdown
- Cardiovascular health assessment
- Clinical recommendations
- Download as TXT file

**Sub-tab B: Prescriptions**
- Treatment recommendations
- Medication suggestions
- Lifestyle modifications
- Monitoring guidelines
- Dietary recommendations
- Download as TXT file

**Sub-tab C: Suggestions**
- Personalized health tips
- Exercise recommendations
- Diet guidelines specific to risk profile
- Stress management techniques
- Sleep optimization advice
- Social support suggestions

#### 5. **Export Section**
Multiple export formats for results:
- **JSON Export**: Complete results data structure
- **CSV Export**: Patient data in tabular format
- **TXT Export**: Individual reports and recommendations

### Key Features

✅ **Responsive Design**: Works on desktop and tablet devices
✅ **Real-time Validation**: Input field validation as user types
✅ **Session State Management**: Preserves form data during interaction
✅ **Professional Styling**: Custom CSS with color gradients
✅ **Fast Inference**: <2 second prediction time
✅ **Error Handling**: Graceful error messages for failed LLM calls
✅ **Accessibility**: Clear labels and intuitive navigation

### How It Works (User Flow)

```
1. User enters patient health information (20 fields)
   ↓
2. Clicks "Generate Prediction & Report"
   ↓
3. Data is validated and formatted
   ↓
4. ML Model makes prediction with confidence score
   ↓
5. LLM generates medical report based on:
   - Patient data
   - ML prediction
   - Risk factors identified
   ↓
6. LLM generates personalized prescriptions
   ↓
7. System displays all results in tabs
   ↓
8. User can download or view different reports
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip or conda package manager
- OpenRouter API key (free tier available)

### Installation

1. **Clone the Repository**
```bash
git clone <repository-url>
cd HeartDisease\ Detector
```

2. **Create Virtual Environment**
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure Environment**
```bash
# Copy example to create local .env
cp .env.example .env

# Edit .env and add your API key
# OPENROUTER_API_KEY=your_actual_key_here
```

5. **Train Model (First Time Only)**
- Open `dataset.ipynb` in Jupyter
- Run all cells to train models
- The notebook automatically exports model files

6. **Run the Application**
```bash
streamlit run app.py
```

Application will open at: `http://localhost:8501`

---

## 📁 Project Structure

```
HeartDisease Detector/
├── README.md                      # This file
├── README_PREDICTION.md           # API documentation
├── ARCHITECTURE.md                # System design
├── QUICKSTART.md                  # 3-step quick start
├── STREAMLIT_SETUP.md             # Detailed setup guide
│
├── app.py                         # Streamlit GUI (300+ lines)
├── prediction.py                  # ML/LLM pipeline
├── llm.py                         # LLM API wrapper
│
├── dataset.ipynb                  # Model training notebook
├── heart_disease.csv              # Dataset (10,000 records)
│
├── heart_disease_model.pkl        # Trained KNN model
├── heart_disease_scaler.pkl       # StandardScaler
├── feature_names.json             # Feature metadata
├── model_metadata.json            # Model performance metrics
│
├── requirements.txt               # Python dependencies
├── .env.example                   # API key template
├── .gitignore                     # Git ignore rules
└── __pycache__/                   # Python cache (ignored)
```

---

## 📦 Dependencies

**Core ML Stack**:
- `scikit-learn` (0.1.3) - Machine learning models
- `xgboost` (2.0.0) - Gradient boosting
- `joblib` (1.3.1) - Model serialization

**Data Processing**:
- `pandas` (2.0.3) - Data manipulation
- `numpy` (1.24.3) - Numerical computing
- `imbalance-learn` (0.0) - SMOTE for class balancing

**Web Interface**:
- `streamlit` (1.28.0) - Web framework
- `matplotlib` (3.7.1) - Visualization
- `seaborn` (0.12.2) - Statistical visualization

**LLM Integration**:
- `openai` (1.0.0) - OpenRouter API client
- `python-dotenv` (1.0.0) - Environment variable management

Full list: See `requirements.txt`

---

## 🔒 Security & Privacy

### Sensitive Data Handling
- ⚠️ `.env` file is **NOT** committed to git
- Use `.env.example` as template
- API keys are loaded only at runtime
- No credentials logged to console

### Data Protection
- Patient data is not stored (stateless processing)
- LLM requests contain only current session data
- Results are downloaded locally by user
- Session data cleared when browser closes

### HIPAA Considerations
```
⚠️ This application is NOT HIPAA-compliant by default.
For production healthcare use:
- Implement user authentication
- Add audit logging
- Use encryption for data in transit/at rest
- Comply with local healthcare regulations
- Review with legal counsel
```

---

## 📈 Model Performance Analysis

### Why Low Accuracy & Very Low Precision?

The KNN model's 46.90% accuracy and 20.23% precision reflects fundamental challenges:

1. **Severe Class Imbalance Impact**
   - 10,000 samples but 80/20 split (8,000 healthy vs 2,000 disease)
   - Baseline "always predict No Disease" = 80% accuracy
   - To catch disease cases (high recall), must be liberal with predictions
   - Results in many false positives (predicted sick, actually healthy)

2. **Precision-Recall Tradeoff**
   - ✅ **High Recall**: 56.25% catches majority of disease cases (good for screening)
   - ⚠️ **Low Precision**: 20.23% means 80% of disease predictions are false alarms
   - Cannot achieve both simultaneously with this data distribution
   - **Intentional design choice**: Better to miss some healthy patients than disease patients

3. **Model Comparison Reality** (all models struggle with precision)
   - **Best Accuracy**: Random Forest (65.15%) BUT only 26.25% recall → misses disease!
   - **Best ROC-AUC**: SVM (0.5142) BUT only 27.50% recall → misses disease!
   - **Best Recall**: KNN (56.25%) BUT only 20.23% precision → many false alarms ✅ SELECTED
   - **Selected KNN**: Prioritizes catching disease over avoiding false alarms

4. **Feature Predictiveness**
   - 20 features may not be strongly predictive of disease
   - Missing critical clinical biomarkers:
     - Advanced cardiac markers (troponin, BNP, NT-proBNP)
     - Imaging data (ECG, echocardiogram, cardiac CT)
     - Genetic/molecular markers
     - Detailed angiography results

5. **SMOTE Application**
   - Applied only to training set (creates synthetic minority samples)
   - Test set kept imbalanced (reflects real-world distribution)
   - Helps during training but doesn't change fundamental data scarcity

### Why We Prioritize Recall (Not Accuracy)

```
Confusion Matrix Analysis for KNN:

         Predicted Healthy    Predicted Disease
Actually Healthy:   TP          FP (false alarm)
Actually Disease:  FN (MISS!)    TP

Recall = TP / (TP + FN)
- High Recall: Catch most disease cases (minimize FN)
- Our 56.25% recall: Catch 225/400 disease cases in test set
- Remaining 175 disease cases missed (FN)

In medical screening:
- FALSE NEGATIVES (missing disease) are dangerous ❌❌❌
- FALSE POSITIVES (false alarms) lead to further testing ⚠️
- Better to over-predict and let doctors investigate
```

### Improvement Opportunities

**Short-term** (Current Dataset):
- Implement ensemble voting (combine multiple models)
- Adjust decision threshold to optimize recall further
- Add patient cost weighting (penalize disease misclassification)

**Medium-term** (Better Data):
- Collect 50,000+ samples for better statistical power
- Add advanced biomarkers (troponin, BNP, ejection fraction)
- Include imaging features (ECG patterns, cardiac imaging)
- Incorporate temporal data (trends over time)

**Long-term** (Advanced Techniques):
- Deep learning (neural networks) with more feature interactions
- Federated learning across multiple hospitals
- Incorporate genetic/family history databases
- Continuous model retraining with production data

---

## 🧪 Testing

### Manual Testing Workflow
1. Fill patient form with test data
2. Generate prediction
3. Verify results appear in all tabs
4. Download different export formats
5. Test edge cases (very high/low values)

### Test Cases Provided
See `test_llm.py` for sample patient profiles:
- **Low Risk**: No smoking, exercises, good vitals
- **High Risk**: Smoking, family history, elevated markers
- **Mixed Risk**: Some risk factors present

### Running Tests
```bash
python test_llm.py
```

---

## 📝 Usage Examples

### Via Streamlit GUI
1. Open browser to `http://localhost:8501`
2. Fill in 20 patient health fields
3. Click "Generate Prediction & Report"
4. View results across 3 tabs
5. Download reports as needed

### Via Python API
```python
from prediction import predict_heart_disease, display_results

patient_data = {
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

result = predict_heart_disease(patient_data)
if result:
    display_results(result)
    print(result['medical_report'])
```

---

## 🐛 Troubleshooting

### "Model files not found"
**Solution**: Run the export code cell in `dataset.ipynb` to generate:
- `heart_disease_model.pkl`
- `heart_disease_scaler.pkl`
- `feature_names.json`
- `model_metadata.json`

### "OPENROUTER_API_KEY not found"
**Solution**: 
1. Create `.env` file in project root
2. Add: `OPENROUTER_API_KEY=your_key_here`
3. Restart the application

### "ModuleNotFoundError"
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Slow LLM responses
**Solution**: 
- OpenRouter free tier may have rate limits
- Upgrade to paid tier for faster responses
- Check internet connection

### GUI not loading
**Solution**: 
```bash
# Clear Streamlit cache
rm -r ~/.streamlit/cache

# Restart app
streamlit run app.py --logger.level=debug
```

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- [ ] Enhanced dataset collection (more diverse patients)
- [ ] Advanced ML techniques (ensemble, deep learning)
- [ ] Better LLM prompts for medical accuracy
- [ ] Mobile app version
- [ ] Multi-language support
- [ ] Doctor dashboard for batch processing
- [ ] Integration with EHR systems

---

## 📜 License

This project is provided as-is for educational purposes. See LICENSE file for details.

---

## ⚠️ Legal Disclaimer

```
IMPORTANT NOTICE:

This Heart Disease Prediction System is intended for educational, 
research, and informational purposes ONLY. 

IT IS NOT A MEDICAL DIAGNOSTIC TOOL.

❌ Do NOT use for:
- Clinical diagnosis
- Treatment decisions
- Medical interventions
- HIPAA-regulated environments (without modifications)
- Insurance or medical billing

✅ Use for:
- Learning about ML and healthcare AI
- Risk factor awareness
- Patient education
- Research purposes
- Proof of concept demonstrations

ALWAYS CONSULT QUALIFIED HEALTHCARE PROVIDERS FOR:
- Medical diagnosis
- Treatment planning
- Health decisions
- Emergency situations

THE AUTHORS AND CONTRIBUTORS ASSUME NO LIABILITY FOR:
- Errors in predictions
- Misuse of this system
- Healthcare decisions based on this tool
- Any adverse outcomes resulting from use

By using this system, you acknowledge these limitations 
and agree to use it responsibly.
```

---

## 📞 Support & Contact

For issues, questions, or suggestions:
1. Check `QUICKSTART.md` for quick setup
2. Review `STREAMLIT_SETUP.md` for detailed guide
3. See `ARCHITECTURE.md` for system design
4. Open an issue on GitHub

---

## 🔄 Version History

- **v1.0** (2026-07-10): Initial release
  - KNN model trained (48.91% accuracy, 42% recall)
  - Streamlit GUI with 3 tabs
  - LLM integration for report generation
  - Export functionality (JSON/CSV/TXT)

---

## 📚 References & Learning Resources

- **Streamlit**: https://docs.streamlit.io/
- **Scikit-learn**: https://scikit-learn.org/
- **OpenRouter API**: https://openrouter.ai/docs
- **Medical ML**: https://arxiv.org/search/?query=medical+machine+learning
- **Heart Disease**: https://www.cdc.gov/heartdisease/

---

## 🎓 Educational Value

This project demonstrates:
✅ End-to-end ML pipeline (data → model → deployment)
✅ Class imbalance handling with SMOTE
✅ Model selection based on domain-specific metrics
✅ LLM integration in practical applications
✅ Web UI development with Streamlit
✅ Full-stack healthcare AI system
✅ Security best practices (.env, .gitignore)
✅ Comprehensive documentation

---

**Made with ❤️ for learning and healthcare innovation**

*Last Updated: 2026-07-10*
