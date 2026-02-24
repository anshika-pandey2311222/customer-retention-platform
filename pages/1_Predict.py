import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.title("🔍 Predict Customer Churn")

st.markdown("Enter customer details to predict churn risk.")

credit_score = st.number_input("Credit Score", 300, 900)
age = st.number_input("Age", 18, 100)
tenure = st.number_input("Tenure", 0, 20)
balance = st.number_input("Balance", 0.0)
products_number = st.number_input("Number of Products", 1, 4)
credit_card = st.selectbox("Has Credit Card?", [0, 1])
active_member = st.selectbox("Active Member?", [0, 1])
estimated_salary = st.number_input("Estimated Salary", 0.0)
country = st.selectbox("Country", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])

if st.button("Predict"):

    input_data = {
        "credit_score": credit_score,
        "age": age,
        "tenure": tenure,
        "balance": balance,
        "products_number": products_number,
        "credit_card": credit_card,
        "active_member": active_member,
        "estimated_salary": estimated_salary,
        "country_Germany": 1 if country == "Germany" else 0,
        "country_Spain": 1 if country == "Spain" else 0,
        "gender_Male": 1 if gender == "Male" else 0
    }

    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=columns, fill_value=0)
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")
    st.write(f"Churn Probability: {probability*100:.2f}%")
