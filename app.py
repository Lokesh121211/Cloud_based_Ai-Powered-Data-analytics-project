import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("financial_planner_model.pkl")
encoder = joblib.load("lifestyle_encoder.pkl")

st.set_page_config(page_title="AI Financial Advisor", page_icon="💰", layout="wide")

st.title("💰 AI Personal Finance Advisor")
st.write("Plan your finances using Machine Learning")

salary = st.number_input("Enter Monthly Salary (₹)", min_value=1000)

lifestyle = st.selectbox(
    "Select Lifestyle",
    ["Basic","Middle","Rich"]
)

if st.button("Generate Financial Plan"):

    life_encoded = encoder.transform([lifestyle])[0]

    user_df = pd.DataFrame({
        "salary":[salary],
        "lifestyle_encoded":[life_encoded]
    })

    prediction = model.predict(user_df)[0]

    rent, food, travel, gym, misc, savings = prediction

    st.subheader("Financial Overview")

    col1,col2,col3 = st.columns(3)
    col1.metric("Rent", f"₹{rent:.0f}")
    col2.metric("Food", f"₹{food:.0f}")
    col3.metric("Travel", f"₹{travel:.0f}")

    col4,col5,col6 = st.columns(3)
    col4.metric("Gym", f"₹{gym:.0f}")
    col5.metric("Misc", f"₹{misc:.0f}")
    col6.metric("Savings", f"₹{savings:.0f}")

    st.subheader("Expense Breakdown")

    expense_df = pd.DataFrame({
        "Category":["Rent","Food","Travel","Gym","Misc"],
        "Amount":[rent,food,travel,gym,misc]
    })

    st.bar_chart(expense_df.set_index("Category"))

    savings_ratio = savings/salary

    st.subheader("AI Advice")

    if savings_ratio < 0.1:
        st.error("Your savings are low.")
    elif savings_ratio < 0.25:
        st.warning("Moderate savings. Consider investments.")
    else:
        st.success("Excellent savings discipline.")