import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import pandas as pd
import joblib

data = pd.read_csv('Stars.csv')

# Use LabelEncoder for Color and Spectral_Class
color_encoder = LabelEncoder()
spectral_encoder = LabelEncoder()

data['Color'] = color_encoder.fit_transform(data['Color'])
data['Spectral_Class'] = spectral_encoder.fit_transform(data['Spectral_Class'])
scaler = StandardScaler()
X = data.drop(['Type'], axis=1)
X = scaler.fit_transform(X)

model = joblib.load('Stars_model.joblib')

st.title("What type of Star is it? ✨")
Temperature = st.slider("Input the Temperature in Kelvin", 1939, 40000)
L = st.slider("Input the Relative Luminosity", 0, 849000)
R = st.slider("What is the Relative Radius", 0.01, 1950.0)
A_M = st.slider("Provide the Absolute Magnitude", -11.9, 20.1)
Color = st.select_slider("Select the General Observable color", [
    'Red', 'Blue White', 'White', 'Yellowish White', 'Blue white',
    'Pale yellow orange', 'Blue', 'Blue-white', 'Whitish', 'yellow-white',
    'Orange', 'White-Yellow', 'white', 'yellowish', 'Yellowish', 'Orange-Red',
    'Blue-White'
])
Spectral_Class = st.select_slider("Select the SMASS Spectral Class", ['M', 'B', 'A', 'F', 'O', 'K', 'G'])

def predict():
    row = np.array([Temperature, L, R, A_M, Color, Spectral_Class])
    X_input = pd.DataFrame([row], columns=['Temperature', 'L', 'R', 'A_M', 'Color', 'Spectral_Class'])
    X_input['Color'] = color_encoder.transform(X_input['Color'])
    X_input['Spectral_Class'] = spectral_encoder.transform(X_input['Spectral_Class'])
    X_input = scaler.transform(X_input)
    prediction = model.predict(X_input)[0]

    star_types = ["Red Dwarf", "Brown Dwarf", "White Dwarf", "Main Sequence", "Super Giants", "Hyper Giants"]
    if prediction >= 0 and prediction < len(star_types):
        st.success(f"Based on your inputs, your STAR is a:\n {star_types[int(prediction)]}")

st.button("Predict", on_click=predict)
