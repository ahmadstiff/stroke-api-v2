import streamlit as st
import numpy as np
import joblib
import os

# Load model
MODEL_PATH = os.path.join("model", "lightgbm_model.pkl")
model = joblib.load(MODEL_PATH)

# Feature order sesuai training
FEATURE_ORDER = [
    'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
    'ever_married', 'Residence_type',
    'gender_Male', 'gender_Other',
    'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private', 'work_type_Self_employed',
    'smoking_status_formerly_smoked', 'smoking_status_never_smoked', 'smoking_status_smokes'
]

# Pilih Bahasa
lang = st.radio("🌐 Choose Language / Pilih Bahasa", ["English", "Bahasa Indonesia"])

# Multi-language labels
TEXT = {
    "English": {
        "title": "🧠 Stroke Prediction App",
        "desc": "Fill out the form below to predict stroke risk.",
        "age": "Age (years)",
        "weight": "Weight (kg)",
        "height": "Height (cm)",
        "married": "Marital Status",
        "gender": "Gender",
        "smoking": "Smoking Status",
        "glucose": "Average Glucose Level",
        "hypertension": "Do you have hypertension?",
        "heart": "Do you have heart disease?",
        "work": "Occupation",
        "residence": "Living Area",
        "submit": "🔍 Predict Stroke",
        "bmi_result": "**Your BMI is:**",
        "result_risk": "❗ Result: The person is **at risk** of stroke.",
        "result_safe": "✅ Result: The person is **not at risk** of stroke.",
        "warn_age": "🚫 Age must be between 1 and 120 years.",
        "warn_weight": "🚫 Weight must be between 10 and 300 kg.",
        "warn_height": "🚫 Height must be between 100 and 250 cm.",
        "married_choices": ["No", "Yes"],
        "gender_choices": ["Female", "Male", "Other"],
        "smoking_choices": ["Unknown", "Former smoker", "Never smoked", "Smokes"],
        "work_choices": ["Children", "Govt job", "Never worked", "Private", "Self employed"],
        "residence_choices": ["Urban", "Rural"],
        "yes": "Yes",
        "no": "No"
    },
    "Bahasa Indonesia": {
        "title": "🧠 Aplikasi Prediksi Stroke",
        "desc": "Isi data di bawah ini untuk memprediksi risiko stroke.",
        "age": "Usia (tahun)",
        "weight": "Berat Badan (kg)",
        "height": "Tinggi Badan (cm)",
        "married": "Status Pernikahan",
        "gender": "Jenis Kelamin",
        "smoking": "Status Merokok",
        "glucose": "Rata-rata Gula Darah",
        "hypertension": "Apakah memiliki hipertensi?",
        "heart": "Apakah memiliki penyakit jantung?",
        "work": "Pekerjaan",
        "residence": "Area Tempat Tinggal",
        "submit": "🔍 Prediksi Stroke",
        "bmi_result": "**BMI Anda:**",
        "result_risk": "❗ Hasil: Pasien **berisiko** terkena stroke.",
        "result_safe": "✅ Hasil: Pasien **tidak berisiko** terkena stroke.",
        "warn_age": "🚫 Usia harus antara 1 sampai 120 tahun.",
        "warn_weight": "🚫 Berat badan harus antara 10 sampai 300 kg.",
        "warn_height": "🚫 Tinggi badan harus antara 100 sampai 250 cm.",
        "married_choices": ["Belum", "Sudah"],
        "gender_choices": ["Perempuan", "Laki-laki", "Lainnya"],
        "smoking_choices": ["Tidak diketahui", "Mantan perokok", "Tidak merokok", "Merokok"],
        "work_choices": ["Anak-anak", "PNS", "Belum pernah kerja", "Swasta", "Wiraswasta"],
        "residence_choices": ["Perkotaan", "Pedesaan"],
        "yes": "Ya",
        "no": "Tidak"
    }
}

T = TEXT[lang]

# Page setup
st.set_page_config(page_title=T["title"], page_icon="🧠", layout="centered")
st.markdown(f"<h1 style='text-align: center; color: #4e8cff;'>{T['title']}</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center;'>{T['desc']}</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Form Input ---
with st.form("stroke_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input(T["age"], min_value=1, max_value=120, value=30)
        weight = st.number_input(T["weight"], min_value=10.0, max_value=300.0, value=60.0)
        married = st.selectbox(T["married"], T["married_choices"])
        gender = st.selectbox(T["gender"], T["gender_choices"])
        smoking = st.selectbox(T["smoking"], T["smoking_choices"])
    with col2:
        height = st.number_input(T["height"], min_value=100.0, max_value=250.0, value=170.0)
        glucose = st.number_input(T["glucose"], min_value=0.0, max_value=300.0, value=100.0)
        hypertension = st.selectbox(T["hypertension"], [T["no"], T["yes"]])
        heart_disease = st.selectbox(T["heart"], [T["no"], T["yes"]])
        work = st.selectbox(T["work"], T["work_choices"])
        residence = st.selectbox(T["residence"], T["residence_choices"])
    
    submitted = st.form_submit_button(T["submit"])

# --- Validasi dan Prediksi ---
if submitted:
    if age < 1 or age > 120:
        st.warning(T["warn_age"])
    elif weight < 10 or weight > 300:
        st.warning(T["warn_weight"])
    elif height < 100 or height > 250:
        st.warning(T["warn_height"])
    else:
        height_m = height / 100
        bmi = weight / (height_m ** 2)
        st.markdown(f"<p style='text-align: center;'>📊 <strong>{T['bmi_result']}</strong> {bmi:.2f}</p>", unsafe_allow_html=True)

        # Encode categorical features
        gender_Male = 1 if gender in ["Male", "Laki-laki"] else 0
        gender_Other = 1 if gender in ["Other", "Lainnya"] else 0

        work_mapping = {
            "Govt job": "work_type_Govt_job", "PNS": "work_type_Govt_job",
            "Never worked": "work_type_Never_worked", "Belum pernah kerja": "work_type_Never_worked",
            "Private": "work_type_Private", "Swasta": "work_type_Private",
            "Self employed": "work_type_Self_employed", "Wiraswasta": "work_type_Self_employed"
        }
        work_features = {key: 0 for key in [
            "work_type_Govt_job", "work_type_Never_worked", "work_type_Private", "work_type_Self_employed"
        ]}
        if work in work_mapping:
            work_features[work_mapping[work]] = 1

        smoking_mapping = {
            "formerly_smoked": "smoking_status_formerly_smoked", "Mantan perokok": "smoking_status_formerly_smoked",
            "never_smoked": "smoking_status_never_smoked", "Tidak merokok": "smoking_status_never_smoked",
            "smokes": "smoking_status_smokes", "Merokok": "smoking_status_smokes"
        }
        smoking_features = {key: 0 for key in [
            "smoking_status_formerly_smoked", "smoking_status_never_smoked", "smoking_status_smokes"
        ]}
        if smoking in smoking_mapping:
            smoking_features[smoking_mapping[smoking]] = 1

        input_data = [
            age,
            1 if hypertension in ["Yes", "Ya"] else 0,
            1 if heart_disease in ["Yes", "Ya"] else 0,
            glucose,
            bmi,
            1 if married in ["Yes", "Sudah"] else 0,
            1 if residence in ["Urban", "Perkotaan"] else 0,
            gender_Male,
            gender_Other,
            *work_features.values(),
            *smoking_features.values()
        ]

        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)[0]

        st.markdown("---")
        if prediction == 1:
            st.error(T["result_risk"])
        else:
            st.success(T["result_safe"])
