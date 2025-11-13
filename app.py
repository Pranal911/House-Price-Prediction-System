import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


model_files = {
    "Linear Regression": "ML_Model/Linear_Regression_model.pkl",
    "Decision Tree": "ML_Model/Decision_Tree_model.pkl",
    "Random Forest": "ML_Model/Random_Forest_model.pkl",
    "KNN": "ML_Model/KNN_model.pkl",
    "SVM": "ML_Model/SVM_model.pkl",
    "Naive Bayes (Classification)": "ML_Model/Naive_Bayes_model.pkl"
}

loaded_models = {}
for name, path in model_files.items():
    if os.path.exists(path):
        loaded_models[name] = joblib.load(path)


st.title(" House Price Prediction System")
st.write("Enter the house details below to predict the price.")

area = st.number_input("Area (sq ft)", min_value=500, max_value=20000, value=7500)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
stories = st.number_input("Number of Stories", min_value=1, max_value=5, value=2)
parking = st.number_input("Parking Spaces", min_value=0, max_value=5, value=1)

mainroad = st.selectbox("Main Road?", ["yes", "no"])
guestroom = st.selectbox("Guest Room?", ["yes", "no"])
basement = st.selectbox("Basement?", ["yes", "no"])
hotwaterheating = st.selectbox("Hot Water Heating?", ["yes", "no"])
airconditioning = st.selectbox("Air Conditioning?", ["yes", "no"])
prefarea = st.selectbox("Preferred Area?", ["yes", "no"])
furnishingstatus = st.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"])

input_data = pd.DataFrame([[
    area, bedrooms, bathrooms, stories, parking,
    mainroad, guestroom, basement, hotwaterheating,
    airconditioning, prefarea, furnishingstatus
]], columns=[
    "area", "bedrooms", "bathrooms", "stories", "parking",
    "mainroad", "guestroom", "basement", "hotwaterheating",
    "airconditioning", "prefarea", "furnishingstatus"
])


st.subheader("Choose a Model for Prediction")

model_choice = st.selectbox("Select Model", list(loaded_models.keys()))

if st.button("Predict"):
    model = loaded_models[model_choice]
    prediction = model.predict(input_data)

    if "Naive Bayes" in model_choice:
        st.success(f" Predicted Price Category: **{prediction[0]}**")
    else:
        st.success(f" Predicted Price: **Rupees {int(prediction[0]):,}**")