# ============================================
# PHASE 1 — IMPORT LIBRARIES
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import joblib


# ============================================
# PHASE 2 — GENERATE SYNTHETIC DATASET (5000 ROWS)
# ============================================

np.random.seed(42)

n = 5000

salary = np.random.randint(15000, 250000, n)
lifestyle = np.random.choice(["Basic","Middle","Rich"], n)

rent=[]
food=[]
travel=[]
gym=[]
misc=[]
total_expense=[]
savings=[]

for i in range(n):

    s = salary[i]
    life = lifestyle[i]

    if life=="Basic":

        r = s*np.random.uniform(0.20,0.25)
        f = s*np.random.uniform(0.20,0.25)
        t = s*np.random.uniform(0.05,0.08)
        g = s*np.random.uniform(0.02,0.04)
        m = s*np.random.uniform(0.10,0.15)

    elif life=="Middle":

        r = s*np.random.uniform(0.25,0.30)
        f = s*np.random.uniform(0.20,0.25)
        t = s*np.random.uniform(0.08,0.12)
        g = s*np.random.uniform(0.04,0.07)
        m = s*np.random.uniform(0.10,0.15)

    else:

        r = s*np.random.uniform(0.30,0.35)
        f = s*np.random.uniform(0.25,0.30)
        t = s*np.random.uniform(0.10,0.15)
        g = s*np.random.uniform(0.05,0.08)
        m = s*np.random.uniform(0.08,0.12)

    expense = r+f+t+g+m
    save = s-expense

    rent.append(r)
    food.append(f)
    travel.append(t)
    gym.append(g)
    misc.append(m)
    total_expense.append(expense)
    savings.append(save)

data = pd.DataFrame({
    "salary":salary,
    "lifestyle":lifestyle,
    "rent":rent,
    "food":food,
    "travel":travel,
    "gym":gym,
    "misc":misc,
    "total_expense":total_expense,
    "savings":savings
})

print("Dataset Shape:",data.shape)

data.to_csv("financial_dataset_5000.csv",index=False)


# ============================================
# PHASE 3 — DATA PREPROCESSING
# ============================================

encoder = LabelEncoder()

data["lifestyle_encoded"] = encoder.fit_transform(data["lifestyle"])


# ============================================
# PHASE 4 — FEATURE ENGINEERING
# ============================================

X = data[["salary","lifestyle_encoded"]]

y = data[["rent","food","travel","gym","misc","savings"]]


# ============================================
# PHASE 5 — TRAIN TEST SPLIT
# ============================================

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)


# ============================================
# PHASE 6 — MACHINE LEARNING MODEL
# ============================================

model = MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=200,
                random_state=42
            )
        )

model.fit(X_train,y_train)


# ============================================
# PHASE 7 — MODEL EVALUATION
# ============================================

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print("\nModel Evaluation")
print("MAE:",mae)
print("R2 Score:",r2)


# ============================================
# PHASE 8 — SAVE MODEL
# ============================================

import os

save_path = r"C:\Users\lokes\Desktop\Financial_AI_Project"

joblib.dump(model, os.path.join(save_path,"financial_planner_model.pkl"))
joblib.dump(encoder, os.path.join(save_path,"lifestyle_encoder.pkl"))

data.to_csv(os.path.join(save_path,"financial_dataset_5000.csv"), index=False)

# ============================================
# PHASE 11 — AI CHAT FINANCIAL ADVISOR
# ============================================

print("\n====== AI CHAT FINANCIAL ADVISOR ======")
print("Type 'exit' anytime to stop.\n")

while True:

    user_query = input("You: ").lower()

    if user_query == "exit":
        print("AI: Thank you for using the Financial Advisor!")
        break


    elif "plan" in user_query or "finance" in user_query:

        salary_input = float(input("Enter your Monthly Salary (₹): "))
        lifestyle_input = input("Choose Lifestyle (Basic/Middle/Rich): ").capitalize()

        life_encoded = encoder.transform([lifestyle_input])[0]

        user_df = pd.DataFrame({
            "salary":[salary_input],
            "lifestyle_encoded":[life_encoded]
        })

        prediction = model.predict(user_df)[0]

        rent_p = prediction[0]
        food_p = prediction[1]
        travel_p = prediction[2]
        gym_p = prediction[3]
        misc_p = prediction[4]
        savings_p = prediction[5]

        total_expense = rent_p+food_p+travel_p+gym_p+misc_p

        print("\nAI Financial Plan")
        print(f"Rent: ₹{rent_p:.2f}")
        print(f"Food: ₹{food_p:.2f}")
        print(f"Travel: ₹{travel_p:.2f}")
        print(f"Gym: ₹{gym_p:.2f}")
        print(f"Misc: ₹{misc_p:.2f}")
        print(f"Savings: ₹{savings_p:.2f}")
        print(f"Total Expense: ₹{total_expense:.2f}")

        savings_ratio = savings_p/salary_input

        print("\nAI Advice:")

        if savings_ratio < 0.1:
            print("Your savings are very low. Reduce discretionary expenses.")

        elif savings_ratio < 0.25:
            print("Your savings are moderate. Consider SIP investments.")

        else:
            print("Excellent savings discipline. Invest in long-term assets.")


    elif "save" in user_query:

        salary_input = float(input("Enter your salary: "))
        lifestyle_input = input("Lifestyle (Basic/Middle/Rich): ").capitalize()

        life_encoded = encoder.transform([lifestyle_input])[0]

        user_df = pd.DataFrame({
            "salary":[salary_input],
            "lifestyle_encoded":[life_encoded]
        })

        prediction = model.predict(user_df)[0]

        savings_p = prediction[5]

        print(f"AI: You can save approximately ₹{savings_p:.2f} per month.")


    elif "expense" in user_query or "breakdown" in user_query:

        salary_input = float(input("Enter your salary: "))
        lifestyle_input = input("Lifestyle (Basic/Middle/Rich): ").capitalize()

        life_encoded = encoder.transform([lifestyle_input])[0]

        user_df = pd.DataFrame({
            "salary":[salary_input],
            "lifestyle_encoded":[life_encoded]
        })

        prediction = model.predict(user_df)[0]

        print("\nAI Expense Breakdown")

        print(f"Rent: ₹{prediction[0]:.2f}")
        print(f"Food: ₹{prediction[1]:.2f}")
        print(f"Travel: ₹{prediction[2]:.2f}")
        print(f"Gym: ₹{prediction[3]:.2f}")
        print(f"Misc: ₹{prediction[4]:.2f}")


    else:

        print("AI: I can help with financial planning.")
        print("Try asking:")
        print(" - Plan my finances")
        print(" - Expense breakdown")
        print(" - How much can I save?")
# ============================================
# PHASE 12 — STREAMLIT WEBSITE INTERFACE
# ============================================

import streamlit as st

def run_web_app():

    st.set_page_config(
        page_title="AI Financial Advisor",
        page_icon="💰",
        layout="wide"
    )

    st.title("💰 AI Personal Finance Advisor")
    st.write("Plan your monthly finances using Machine Learning")

    st.sidebar.header("User Inputs")

    salary = st.sidebar.number_input(
        "Enter Monthly Salary (₹)",
        min_value=1000
    )

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

        st.subheader("AI Financial Advice")

        if savings_ratio < 0.1:
            st.error("⚠ Your savings are very low. Try reducing discretionary expenses.")

        elif savings_ratio < 0.25:
            st.warning("⚡ Moderate savings. Consider SIP investments.")

        else:
            st.success("🚀 Excellent savings discipline. Consider long-term investments.")


if __name__ == "__main__":
    pass