import streamlit as st
import pandas as pd
import pickle

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Churn Prediction", page_icon="📊", layout="wide")

st.title("📊 Customer Churn Prediction System")
st.markdown("Predict whether a customer is likely to churn based on their profile.")

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("📝 Enter Customer Details")

credit_score = st.sidebar.number_input("Credit Score", 300, 900)
age = st.sidebar.number_input("Age", 18, 100)
tenure = st.sidebar.number_input("Tenure (Years with Bank)", 0, 20)

balance = st.sidebar.number_input("Balance", 0.0)
products_number = st.sidebar.number_input("Number of Products", 1, 4)

credit_card = st.sidebar.selectbox("Has Credit Card?", [0, 1])
active_member = st.sidebar.selectbox("Is Active Member?", [0, 1])

estimated_salary = st.sidebar.number_input("Estimated Salary", 0.0)

country = st.sidebar.selectbox("Country", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

# ---------------- PREDICTION ----------------
if st.sidebar.button("🔍 Predict Churn"):

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

    # ---------------- RESULT DISPLAY ----------------
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is likely to stay")

    st.write(f"Churn Probability: **{probability:.2f}**")

    st.progress(float(probability))

    if probability > 0.7:
        st.warning("High Risk: Consider retention strategies like discounts or loyalty offers.")