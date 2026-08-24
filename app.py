import streamlit as st
import numpy as np
import pickle

try:
    clf = pickle.load(open("xgb_classifier.pkl", 'rb'))
    reg = pickle.load(open("xgb_regressor.pkl", 'rb'))

except FileNotFoundError:
    st.error("Model files (xgb_classifier.pkl or xgb_regressor.pkl) not found. Please ensure they are in the same directory as this app.py file.")
    st.stop()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

#The features
feature_limits = {
    'PackVoltage_V': (200, 450),            # 200V to 450V
    'CellVoltage_V': (3.0, 4.5),            # 3V to 4.5V per cell
    'ChargeCurrent_A': (0, 300),            # Charging current 0 to 300 A
    'SOC_%': (0, 100),                      # State of charge 0 to 100%
    'MaxTemp_C': (0, 90),                   # Max temp 0 to 90 °C
    'AvgTemp_C': (0, 70),                   # Average temp 0 to 70 °C
    'InternalResistance_mOhm': (0, 100),    # Internal resistance 0 to 100 milliohms
    'Pressure_kPa': (90, 120)               # Pressure between 90 to 120 kPa
}

st.set_page_config(
    page_title="Battery Thermal Runaway Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        border-radius: 8px;
        padding: 10px 24px;
        margin-top: 20px;
    }
    .stTextInput>div>input {
        font-size: 16px;
        padding: 8px;
        border: 2px solid #4CAF50;
        border-radius: 6px;
        transition: border-color 0.3s;
    }
    .stTextInput>div>input:focus {
        border-color: #388E3C;
        outline: none;
    }
    .title {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: bold;
        color: #2E7D32;
    }
    .result-high {
        background-color: #FFCDD2;
        color: #C62828;
        padding: 15px;
        border-radius: 10px;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
    }
    .result-low {
        background-color: #C8E6C9;
        color: #2E7D32;
        padding: 15px;
        border-radius: 10px;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
    }
    .subheader-style {
        color: #388E3C;
        font-weight: 600;
        font-size: 22px;
    }
    .warning {
        color: #D84315;
        font-weight: 600;
        font-size: 16px;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<h1 class="title">EV Battery Thermal Runaway Risk Predictor</h1>', unsafe_allow_html=True)
st.markdown("---")

st.header("Input Battery Parameters")

inputs = {}
errors = []
empty_inputs = False

for feature, (low, high) in feature_limits.items():

    display_feature = feature.replace('_', ' ').replace('z', ' (V)').replace('z', ' (A)').replace('z', ' (°C)').replace('mOhm', ' (mOhm)').replace('kPa', ' (kPa)').replace('%', ' (%)')

    while True:
        val = st.text_input(f"Enter {display_feature} (Range: {low} – {high}):", value="0.0", key=feature)
        try:
            fval = float(val)
            if fval == 0.0:
                empty_inputs = True
            elif not (low <= fval <= high):
                st.warning(f"{feature} should be between {low} and {high}.")
                continue
            inputs[feature] = fval
            break
        except ValueError:
            st.warning(f"Please enter a numeric value for {feature}.")

if st.button("Predict Risk & Probability"):
    if empty_inputs:
        st.error("Prediction cannot be performed because some inputs are zero. Please input valid values within specified ranges.")
    else:
        
        input_array = np.array([inputs[feat] for feat in feature_limits.keys()]).reshape(1, -1)
        
        risk_class = clf.predict(input_array)[0]
        risk_prob = reg.predict(input_array)[0]
        
        risk_status = "High Risk of Thermal Runaway" if risk_class == 1 else "Low Risk"

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown('<div class="subheader-style">Classification Result</div>', unsafe_allow_html=True)
            if risk_class == 1:
                st.markdown(f'<div class="result-high">{risk_status}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-low">{risk_status}</div>', unsafe_allow_html=True)
            st.write(f"Probability of Risk (from regressor): **{risk_prob:.2f}**")
        with col2:
            st.markdown('<div class="subheader-style">Regression Output</div>', unsafe_allow_html=True)
            st.write(f"Estimated Probability of Thermal Runaway:")
            st.markdown(f"<h2 style='color:#2E7D32'>{risk_prob:.2f}</h2>", unsafe_allow_html=True)