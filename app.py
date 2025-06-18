from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)  # ⬅️ Aktifkan CORS

# Load model
MODEL_PATH = os.path.join("model", "lightgbm_model.pkl")
model = joblib.load(MODEL_PATH)

# Define expected feature order (harus sesuai dengan training)
FEATURE_ORDER = [
    'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
    'ever_married', 'Residence_type',
    'gender_Male', 'gender_Other',
    'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private', 'work_type_Self_employed',
    'smoking_status_formerly_smoked', 'smoking_status_never_smoked', 'smoking_status_smokes'
]

@app.route('/')
def index():
    return "Stroke Prediction API - Flask with CORS"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_values = [data.get(feature, 0) for feature in FEATURE_ORDER]
        input_array = np.array(input_values).reshape(1, -1)

        prediction = model.predict(input_array)[0]
        return jsonify({
            "stroke_risk": int(prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)
