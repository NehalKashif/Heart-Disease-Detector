import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def generate_report(patient_data, prediction):

    prompt = f"""
You are a medical assistant.

Prediction:
{"Heart Disease Detected" if prediction else "No Heart Disease"}

Patient Information:
{patient_data}

Generate:
1. Simple explanation
2. Possible risk factors
3. Lifestyle recommendations
4. Suggest consulting a doctor.
5. Mention that this is NOT a diagnosis.
"""

    response = client.chat.completions.create(
        model="meta-llama/llama-3.1-8b-instruct",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    # Test code - only runs if this file is executed directly
    result = generate_report({
            "age": "55",
            "gender": "Male",
            "chest_pain_type": "Asymptomatic",
            "blood_pressure": "140/90",
            "cholesterol": "250 mg/dL",
            "fasting_blood_sugar": "Yes",
            "resting_ecg": "ST-T Wave Abnormality",
            "max_heart_rate": "110 bpm",
            "exercise_induced_angina": "Yes",
            "st_depression": "1.5",
            "st_slope": "Flat",
            "major_vessels": "1 vessel affected"
        }, "1")
    print(result)   