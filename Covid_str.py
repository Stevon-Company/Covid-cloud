
import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="COVID")
st.title("🦠 Covid Test Prediction")



filename = 'Covid_model.pkl'
with open(filename, 'rb') as file:
    Model = pickle.load(file)

st.header("Input Features")

Gender = st.selectbox("Gender", ["Female", "Male"])
Gender_mapping = {
    'Female': 0,
    'Male': 1}
G_encoded = Gender_mapping[Gender]

fever = st.text_input("Fever")

cough = st.selectbox("Cough", ["Mild", "Strong"])
C_mapping = {
    'Mild': 0,
    'Strong': 1}
C_encoded = C_mapping[cough]

city = st.selectbox("City", ["Kolkata", "Bangalore","Delhi","Mumbai"])
city_mapping = {
    'Kolkata': 0,
    'Bangalore': 1,
    "Delhi":2,
    "Mumbai":3}
city_encoded = city_mapping[city]

age=st.text_input("Age")


test_input = {
    "age": [age],

    "fever": [fever],

    "gender_encoded": [G_encoded],
    "cough_encoded": [C_encoded],
    "city_encoded": [city_encoded]
}

test_df = pd.DataFrame(test_input)

# Prediction
if st.button("Predict"):
    prediction = Model.predict(test_df)
    prediction_proba = Model.predict_proba(test_df)

    st.subheader("Prediction")
    if int(prediction[0])==0:
        st.markdown("### Patient has no COVID")
        st.success("🟢 COVID NEGATIVE")

    else:
        st.markdown("### Patient has COVID-19!")
        st.warning("🔴 COVID POSITIVE !!")

    st.write("Prediction probabilities:")
    st.write(prediction_proba)
