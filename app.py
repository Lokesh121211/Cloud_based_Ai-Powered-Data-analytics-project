import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

# ---------------------------------
# Load Dataset
# ---------------------------------

data = pd.read_csv("financial_dataset_5000.csv")

# ---------------------------------
# Preprocess
# ---------------------------------

encoder = LabelEncoder()
data["lifestyle_encoded"] = encoder.fit_transform(data["lifestyle"])

X = data[["salary","lifestyle_encoded"]]
y = data[["rent","food","travel","gym","misc","savings"]]

# ---------------------------------
# Train Model
# ---------------------------------

model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42))
model.fit(X,y)

# ---------------------------------
# Streamlit UI
# ---------------------------------

st.set_page_config(page_title="AI Financial Advisor", page_icon="💰")

st.title("💰 AI Personal Finance Advisor")

st.sidebar.header("User Inputs")

salary = st.sidebar.number_input("Enter Monthly Salary (₹)", min_value=1000)

lifestyle = st.sidebar.selectbox(
    "Select Lifestyle",
    ["Basic","Middle","Rich"]
)

if st.sidebar.button("Generate Financial Plan"):

    life_encoded = encoder.transform([lifestyle])[0]

    user_df = pd.DataFrame({
        "salary":[salary],
        "lifestyle_encoded":[life_encoded]
    })

    prediction = model.predict(user_df)[0]

    rent = prediction[0]
    food = prediction[1]
    travel = prediction[2]
    gym = prediction[3]
    misc = prediction[4]
    savings = prediction[5]

    total_expense = rent + food + travel + gym + misc

    st.subheader("Financial Breakdown")

    col1,col2,col3 = st.columns(3)

    col1.metric("Rent", f"₹{rent:.0f}")
    col2.metric("Food", f"₹{food:.0f}")
    col3.metric("Travel", f"₹{travel:.0f}")

    col4,col5,col6 = st.columns(3)

    col4.metric("Gym", f"₹{gym:.0f}")
    col5.metric("Misc", f"₹{misc:.0f}")
    col6.metric("Savings", f"₹{savings:.0f}")

    st.subheader("Expense Chart")

    expense_df = pd.DataFrame({
        "Category":["Rent","Food","Travel","Gym","Misc"],
        "Amount":[rent,food,travel,gym,misc]
    })

    st.bar_chart(expense_df.set_index("Category"))

    savings_ratio = savings/salary

    st.subheader("AI Financial Advice")

    if savings_ratio < 0.1:
        st.error("Your savings are very low. Reduce unnecessary expenses.")

    elif savings_ratio < 0.25:
        st.warning("Moderate savings. Consider investing in SIPs.")

    else:
        st.success("Excellent savings! Consider long-term investments.")
