import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest

# ------------------ Page Configuration ------------------
st.set_page_config(page_title="Intelligent Process Automation Dashboard", layout="wide")

st.title("Intelligent Process Automation (IPA) Dashboard")
st.markdown("""
This dashboard demonstrates how AI can automate routine approval workflows,
predict task completion timelines, and detect high-risk or anomalous requests.
""")

# ------------------ Task Input Section ------------------
st.subheader("Task Entry")

col1, col2, col3 = st.columns(3)

with col1:
    t1_type = st.selectbox("Task 1 Type", ["Approval", "Purchase", "Compliance"])
    t1_priority = st.selectbox("Task 1 Priority", ["Low", "Medium", "High"])
    t1_amount = st.number_input("Task 1 Amount ($)", min_value=0, value=500)

with col2:
    t2_type = st.selectbox("Task 2 Type", ["Approval", "Purchase", "Compliance"])
    t2_priority = st.selectbox("Task 2 Priority", ["Low", "Medium", "High"])
    t2_amount = st.number_input("Task 2 Amount ($)", min_value=0, value=1200)

with col3:
    t3_type = st.selectbox("Task 3 Type", ["Approval", "Purchase", "Compliance"])
    t3_priority = st.selectbox("Task 3 Priority", ["Low", "Medium", "High"])
    t3_amount = st.number_input("Task 3 Amount ($)", min_value=0, value=800)

# ------------------ Create DataFrame ------------------
tasks = [
    {"Task ID": 1, "Type": t1_type, "Priority": t1_priority, "Amount": t1_amount},
    {"Task ID": 2, "Type": t2_type, "Priority": t2_priority, "Amount": t2_amount},
    {"Task ID": 3, "Type": t3_type, "Priority": t3_priority, "Amount": t3_amount},
]

df = pd.DataFrame(tasks)

# ------------------ Rule-Based Automation ------------------
def auto_approve(row):
    if row["Priority"] == "Low" or row["Amount"] <= 1000:
        return "Approved"
    return "Pending Review"

df["Automation Status"] = df.apply(auto_approve, axis=1)

# ------------------ Predictive Analytics ------------------
historical_amounts = np.array([500, 800, 400, 1500, 1000, 2000, 600, 700, 1200, 900]).reshape(-1, 1)
historical_days = np.array([2, 3, 2, 5, 4, 6, 3, 2, 4, 5])

model = LinearRegression()
model.fit(historical_amounts, historical_days)

df["Predicted Completion (Days)"] = df["Amount"].apply(
    lambda x: max(1, int(model.predict([[x]])[0]))
)

# ------------------ Anomaly Detection ------------------
iso = IsolationForest(contamination=0.2, random_state=42)
df["Anomaly Flag"] = iso.fit_predict(df[["Amount", "Predicted Completion (Days)"]])

anomalies = df[df["Anomaly Flag"] == -1]

# ------------------ KPI Section ------------------
st.subheader("Key Performance Indicators")

k1, k2, k3, k4 = st.columns(4)

k1.metric("Total Tasks", len(df))
k2.metric("Auto-Approved", len(df[df["Automation Status"] == "Approved"]))
k3.metric("Pending Review", len(df[df["Automation Status"] == "Pending Review"]))
k4.metric("Flagged as High-Risk", len(anomalies))

# ------------------ AI Recommendations ------------------
st.subheader("AI Recommendations")

for _, row in df.iterrows():
    if row["Automation Status"] == "Approved":
        st.success(f"Task {row['Task ID']} can be processed automatically.")
    elif row["Anomaly Flag"] == -1:
        st.warning(f"Task {row['Task ID']} appears high-risk and requires manual verification.")
    else:
        st.info(f"Task {row['Task ID']} should be assigned to a human approver.")

# ------------------ Visualization ------------------
st.subheader("Task Analysis Visualization")

fig, ax = plt.subplots(figsize=(8, 4))

colors = ["red" if x == -1 else "blue" for x in df["Anomaly Flag"]]

ax.scatter(df["Task ID"], df["Amount"], c=colors, s=120)

for i in range(len(df)):
    ax.text(
        df["Task ID"][i] + 0.05,
        df["Amount"][i] + 20,
        f"{df['Predicted Completion (Days)'][i]}d"
    )

ax.set_xlabel("Task ID")
ax.set_ylabel("Amount ($)")
ax.set_title("Task Amount vs Predicted Completion Time")
ax.grid(True)

st.pyplot(fig)

# ------------------ Show High-Risk Tasks ------------------
if not anomalies.empty:
    st.subheader("High-Risk Tasks Identified")
    st.dataframe(
        anomalies[
            ["Task ID", "Type", "Priority", "Amount", "Predicted Completion (Days)"]
        ]
    )