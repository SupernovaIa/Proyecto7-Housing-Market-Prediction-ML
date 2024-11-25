import streamlit as st
import pandas as pd
import pickle

from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from src.support_st import *

# Config
st.set_page_config(
    page_title="Las casas de David",
    page_icon="🏠",
    layout="centered",
)

# Title and description
st.title("🏠 House price prediction using ML 🔮")
st.write("Use this app to predict future (in terms of real estate) 🚀")

# Mostrar una imagen llamativa
st.image(
    "https://cdn.midjourney.com/4f7d8a55-1d31-4e80-b726-26050bde3bd8/0_1.png",
    caption="Your next investment is here"
)

# Load model
model = load_models()

municipalities, types, provinces = load_options()


# Forms
st.header("🔧 Features")
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)
col5, col6 = st.columns(2)

with col1:
    propertyType = st.selectbox("House Type", types, help="Select house type.")
    rooms = st.selectbox("Number of rooms", [0, 1, 2, 3, 4, 5, 6],help="Select number of rooms.")

with col2:
    size = st.number_input("Size in sq meters", min_value=10, max_value=300, value=80, step=10, help="Select the size of your house.")
    bathrooms = st.selectbox("Number of bathrooms", [1, 2, 3],help="Select number of bathrooms.")

with col3:
    floor = st.selectbox("Floor", ['ss', 'st', 'bj', 'en', '1', '2', '3', '4', '5', '6', '7', '8', 'unknown'], help="Select floor.")
    municipality = st.selectbox("Municipality", municipalities, help="Select municipality.")

with col4:
    hasLift = st.selectbox("Has lift?", ['Yes', 'No'], help="Select if you want lift.")

    if hasLift == 'Yes':
        hasLift = True
    else:
        hasLift = False

    exterior = st.selectbox("Exterior", ['Yes', 'No'],help="Select if you want an exterior house.")

    if exterior == 'Yes':
        exterior = True
    else:
        exterior = False

with col5:
    province = st.selectbox("Province", provinces, help="Select province.")

with col6:
    distance = st.number_input("Distance",  min_value=0, max_value=60000, value=1000, step=500, help="Select how far you want to be from center (m).")


# Prediction
if st.button("💡 Predict price"):

    pred = get_prediction(model, propertyType, size, exterior, rooms, bathrooms, distance, floor, municipality, province, hasLift, numPhotos=20)
    # Show results
    st.success(f"💵 Expected price is: {round(pred[0],2)} €")
    st.balloons()