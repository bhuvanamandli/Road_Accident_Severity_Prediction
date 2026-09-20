import base64
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model


st.set_page_config(
    page_title="Road Accident Severity Prediction",
    page_icon="🚧"
)

st.title("Road Accident Severity Prediction")


# Load files
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

model = load_model("architechture.h5")

try:
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    class_names = label_encoder.classes_
except FileNotFoundError:
    class_names = ["Fatal", "Serious", "Slight"]
    
def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

img = get_base64("accident_img.jpg")

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{img}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------
# Date & Time
# --------------------------------

st.subheader("📅 Date & Time")

col1, col2 = st.columns(2)

with col1:
    month = st.selectbox(
        "Month",
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )

with col2:
    day_of_week = st.selectbox(
        "Day of Week",
        ["Monday", "Tuesday", "Wednesday", "Thursday",
         "Friday", "Saturday", "Sunday"]
    )

col1, col2 = st.columns(2)

with col1:
    year = st.number_input(
        "Year",
        2000,
        2030,
        2022
    )

with col2:
    hour = st.slider(
        "Hour",
        0,
        23,
        12
    )

urban_or_rural = st.selectbox(
    "Urban or Rural Area",
    ["Urban", "Rural"]
)


# --------------------------------
# Road Details
# --------------------------------

st.subheader("🛣️ Road Details")

col1, col2 = st.columns(2)

with col1:
    road_type = st.selectbox(
        "Road Type",
        [
            "Single carriageway",
            "Dual carriageway",
            "Roundabout",
            "One way street",
            "Slip road",
            "Unknown"
        ]
    )

with col2:
    road_surface_conditions = st.selectbox(
        "Road Surface Conditions",
        [
            "Dry",
            "Wet or damp",
            "Snow",
            "Frost or ice",
            "Flood over 3cm. deep"
        ]
    )

col1, col2 = st.columns(2)

with col1:
    light_conditions = st.selectbox(
        "Light Conditions",
        [
            "Daylight",
            "Darkness - lights lit",
            "Darkness - lights unlit",
            "Darkness - no lighting",
            "Darkness - lighting unknown"
        ]
    )

with col2:
    junction_detail = st.selectbox(
        "Junction Detail",
        [
            "Not at junction or within 20 metres",
            "Roundabout",
            "Mini-roundabout",
            "T or staggered junction",
            "Crossroads",
            "More than 4 arms (not roundabout)",
            "Private drive or entrance",
            "Other junction",
            "Slip road"
        ]
    )

col1, col2 = st.columns(2)

with col1:
    junction_control = st.selectbox(
        "Junction Control",
        [
            "Not at junction or within 20 metres",
            "Give way or uncontrolled",
            "Auto traffic signal",
            "Stop sign",
            "Authorised person",
            "Data missing or out of range"
        ]
    )

with col2:
    speed_limit = st.selectbox(
        "Speed Limit",
        [10, 15, 20, 30, 40, 50, 60, 70]
    )


# --------------------------------
# Vehicle & Weather
# --------------------------------

st.subheader("🚗 Vehicle & Weather")

col1, col2 = st.columns(2)

with col1:
    vehicle_type = st.selectbox(
        "Vehicle Type",
        [
            "Car",
            "Motorcycle 50cc and under",
            "Motorcycle 125cc and under",
            "Motorcycle over 125cc and up to 500cc",
            "Motorcycle over 500cc",
            "Taxi/Private hire car",
            "Minibus (8 - 16 passenger seats)",
            "Bus or coach (17 or more pass seats)",
            "Goods over 3.5t. and under 7.5t",
            "Goods 7.5 tonnes mgw and over",
            "Van / Goods 3.5 tonnes mgw or under",
            "Agricultural vehicle",
            "Pedal cycle",
            "Ridden horse",
            "Other vehicle"
        ]
    )

with col2:
    weather_conditions = st.selectbox(
        "Weather Conditions",
        [
            "Fine no high winds",
            "Raining no high winds",
            "Fine + high winds",
            "Raining + high winds",
            "Snowing no high winds",
            "Snowing + high winds",
            "Fog or mist",
            "Other"
        ]
    )

col1, col2 = st.columns(2)

with col1:
    number_of_vehicles = st.number_input(
        "Number of Vehicles",
        1,
        32,
        2
    )

with col2:
    number_of_casualties = st.number_input(
        "Number of Casualties",
        1,
        48,
        1
    )


# --------------------------------
# Prediction
# --------------------------------

if st.button("🚦 Predict Severity"):

    data = pd.DataFrame([{
        "Month": month,
        "Day_of_Week": day_of_week,
        "Year": year,
        "Junction_Control": junction_control,
        "Junction_Detail": junction_detail,
        "Light_Conditions": light_conditions,
        "Number_of_Casualties": number_of_casualties,
        "Number_of_Vehicles": number_of_vehicles,
        "Road_Surface_Conditions": road_surface_conditions,
        "Road_Type": road_type,
        "Speed_limit": speed_limit,
        "Urban_or_Rural_Area": urban_or_rural,
        "Weather_Conditions": weather_conditions,
        "Vehicle_Type": vehicle_type,
        "Hour": hour
    }])

    # Preprocess
    X = preprocessor.transform(data)

    # Prediction
    probabilities = model.predict(X, verbose=0)[0]

    index = np.argmax(probabilities)
    prediction = class_names[index]
    confidence = probabilities[index] * 100

    # Result
    st.success(
        f"🚦 Prediction: {prediction} ({confidence:.2f}%)"
    )

    # Class probabilities
    st.write("### Class Probabilities")

    result = pd.DataFrame({
        "Severity": class_names,
        "Probability (%)": probabilities * 100
    })

    st.dataframe(
        result,
        hide_index=True
    )
