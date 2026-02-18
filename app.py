import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest

# ------------------ Page Config ------------------
st.set_page_config(page_title="AI Decision Dashboard", layout="wide")
st.title("AI-Powered Business Decision Support System")
st.markdown("### Enter monthly sales to see predictions, anomalies, and recommendations.")

# ------------------ Manual Entry ------------------
st.markdown("### Enter Monthly Sales (6 months)")

col1, col2, col3 = st.columns(3)

with col1:
    m1 = st.number_input("Month 1", value=100)
    m2 = st.number_input("Month 2", value=130)
with col2:
    m3 = st.number_input("Month 3", value=170)
    m4 = st.number_input("Month 4", value=120)
with col3:
    m5 = st.number_input("Month 5", value=150)
    m6 = st.number_input("Month 6", value=300)

sales = [m1, m2, m3, m4, m5, m6]
months = list(range(1, len(sales)+1))

# ------------------ DataFrame ------------------
df = pd.DataFrame({"Month": months, "Sales": sales})
st.markdown("### Sales Overview")
st.dataframe(df, use_container_width=True)

# ------------------ AI Prediction ------------------
X = np.array(months).reshape(-1, 1)
y = np.array(sales)

model = LinearRegression()
model.fit(X, y)

next_month = np.array([[len(sales) + 1]])
next_prediction = int(model.predict(next_month)[0])

# ------------------ Anomaly Detection ------------------
iso = IsolationForest(contamination=0.1, random_state=42)
df['Anomaly'] = iso.fit_predict(df[['Sales']])
anomalies = df[df['Anomaly'] == -1]

# ------------------ Trend Logic (based on last 2 months) ------------------
if sales[-1] >= sales[-2]:
    trend = "Positive"
else:
    trend = "Negative"

# ------------------ KPIs ------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Predicted Next Month Sales", next_prediction)
with col2:
    volatility = np.std(sales)
    risk_score = int((volatility / np.mean(sales)) * 100)
    st.metric("Risk Score (%)", risk_score)
with col3:
    st.metric("Trend Direction", trend)

# ------------------ Recommendation ------------------
st.markdown("### AI Recommendation")
if trend == "Positive":
    st.success("Sales trend is increasing. Expand inventory and marketing.")
else:
    st.warning("Sales trend is declining. Optimize costs and consider promotions.")

# ------------------ Profit Simulation ------------------
st.markdown("### Profit Simulation")
price = st.number_input("Selling Price per Unit", value=50)
cost = st.number_input("Cost per Unit", value=30)
profit = (price - cost) * next_prediction
st.metric("Projected Profit (Next Month)", f"${profit}")

# ------------------ Sales Graph ------------------
st.markdown("### Sales Trend with Anomalies")
plt.figure()
plt.plot(months, sales, marker='o', label="Sales")
if not anomalies.empty:
    plt.scatter(anomalies["Month"], anomalies["Sales"], color='red', label="Anomaly", s=100)
plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Monthly Sales Trend")
plt.legend()
st.pyplot(plt)

# ------------------ Show anomalies table ------------------
if not anomalies.empty:
    st.markdown("### Detected Anomalies")
    st.table(anomalies[["Month", "Sales"]])