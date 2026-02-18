import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Smart Decision Dashboard", layout="wide")

st.title("Smart Business Decision Support System")

st.markdown("### Choose how you want to provide sales data")

option = st.radio("Select Input Method:", ["Manual Entry", "Upload CSV"])

# ------------------ MANUAL ENTRY ------------------

if option == "Manual Entry":

    st.markdown("### Enter Monthly Sales")

    col1, col2, col3 = st.columns(3)

    with col1:
        m1 = st.number_input("Month 1", value=100)
        m2 = st.number_input("Month 2", value=120)

    with col2:
        m3 = st.number_input("Month 3", value=130)
        m4 = st.number_input("Month 4", value=150)

    with col3:
        m5 = st.number_input("Month 5", value=170)
        m6 = st.number_input("Month 6", value=200)

    sales = [m1, m2, m3, m4, m5, m6]

# ------------------ CSV UPLOAD ------------------

else:

    uploaded_file = st.file_uploader("Upload CSV (Columns: Month, Sales)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        sales = list(df["Sales"])
    else:
        st.stop()

# ------------------ ANALYSIS ------------------

months = list(range(1, len(sales)+1))

df = pd.DataFrame({
    "Month": months,
    "Sales": sales
})

st.markdown("### Sales Overview")
st.dataframe(df, use_container_width=True)

# Growth Calculation
growth_rates = []
for i in range(1, len(sales)):
    growth = (sales[i] - sales[i-1]) / sales[i-1]
    growth_rates.append(growth)

avg_growth = np.mean(growth_rates)
next_prediction = int(sales[-1] * (1 + avg_growth))

# KPIs
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Predicted Next Month Sales", next_prediction)

with col2:
    volatility = np.std(sales)
    risk_score = int((volatility / np.mean(sales)) * 100)
    st.metric("Risk Score (%)", risk_score)

with col3:
    trend = "Positive" if avg_growth > 0 else "Negative"
    st.metric("Trend Direction", trend)

# Recommendation
st.markdown("### AI Recommendation")

if avg_growth > 0:
    st.success("Sales trend is increasing. Consider expanding inventory and marketing.")
else:
    st.warning("Sales trend is declining. Focus on cost optimization and promotional offers.")

# Profit Simulation
st.markdown("### Profit Simulation")

price = st.number_input("Selling Price per Unit", value=50)
cost = st.number_input("Cost per Unit", value=30)

profit = (price - cost) * next_prediction

st.metric("Projected Profit (Next Month)", f"${profit}")

# Graph
st.markdown("### Sales Trend Analysis")

plt.figure()
plt.plot(months, sales)
plt.xlabel("Month")
plt.ylabel("Sales")
st.pyplot(plt)
