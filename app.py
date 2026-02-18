import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Smart Decision Dashboard", layout="wide")

st.title("Smart Business Decision Support System")

st.markdown("Enter your monthly sales data below.")

sales_data = st.text_area("Enter 6 months sales (comma separated)", "100,120,130,150,170,200")

if sales_data:
    sales = list(map(int, sales_data.split(",")))
    months = list(range(1, len(sales)+1))

    df = pd.DataFrame({
        "Month": months,
        "Sales": sales
    })

    st.subheader("Sales Overview")
    st.dataframe(df)

    growth_rates = []
    for i in range(1, len(sales)):
        growth = (sales[i] - sales[i-1]) / sales[i-1]
        growth_rates.append(growth)

    avg_growth = np.mean(growth_rates)
    next_prediction = int(sales[-1] * (1 + avg_growth))

    st.subheader("Next Month Prediction")
    st.metric("Predicted Sales", next_prediction)

    volatility = np.std(sales)
    risk_score = int((volatility / np.mean(sales)) * 100)

    st.subheader("Business Risk Level")
    st.progress(min(risk_score, 100))
    st.write(f"Risk Score: {risk_score}%")

    st.subheader("Recommendation")
    if avg_growth > 0:
        st.success("Sales trend is positive. Increase production and marketing.")
    else:
        st.warning("Sales trend is declining. Optimize costs and consider discounts.")

    st.subheader("Profit Simulation")
    price = st.number_input("Selling Price per Unit", value=50)
    cost = st.number_input("Cost per Unit", value=30)

    profit = (price - cost) * next_prediction

    st.metric("Projected Profit Next Month", f"${profit}")

    st.subheader("Sales Trend Graph")
    plt.plot(months, sales)
    plt.xlabel("Month")
    plt.ylabel("Sales")
    st.pyplot(plt)

else:
    st.info("Please enter sales data.")
