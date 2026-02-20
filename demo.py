import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest

# ------------------ Page Setup ------------------
st.set_page_config(page_title="Smart IPA Dashboard", layout="wide")
st.title("AI-Powered Intelligent Process Automation Dashboard")
st.markdown("### Manage and automate routine approval tasks with AI predictions and anomaly detection.")

# ------------------ Manual Task Entry ------------------
st.markdown("### Enter New Task Requests (6 tasks max)")

col1, col2, col3 = st.columns(3)
with col1:
    t1_type = st.selectbox("Task 1 Type", ["Approval", "Purchase", "Compliance"])
    t1_priority = st.selectbox("Task 1 Priority", ["Low", "Medium", "High"])
    t1_amount = st.number_input("Task 1 Amount ($)", value=500)
with col2:
    t2_type = st.selectbox("Task 2 Type", ["Approval", "Purchase", "Compliance"])
    t2_priority = st.selectbox("Task 2 Priority", ["Low", "Medium", "High"])
    t2_amount = st.number_input("Task 2 Amount ($)", value=1200)
with col3:
    t3_type = st.selectbox("Task 3 Type", ["Approval", "Purchase", "Compliance"])
    t3_priority = st.selectbox("Task 3 Priority", ["Low", "Medium", "High"])
    t3_amount = st.number_input("Task 3 Amount ($)", value=800)

# You can expand for more tasks easily
tasks = [
    {"Task ID": 1, "Type": t1_type, "Priority": t1_priority, "Amount": t1_amount, "Status": "Pending"},
    {"Task ID": 2, "Type": t2_type, "Priority": t2_priority, "Amount": t2_amount, "Status": "Pending"},
    {"Task ID": 3, "Type": t3_type, "Priority": t3_priority, "Amount": t3_amount, "Status": "Pending"},
]

df = pd.DataFrame(tasks)

# ------------------ Auto-Approval Logic ------------------
def auto_approve(row):
    # Simple rule: auto-approve low priority or small amount tasks
    if row["Priority"] == "Low" or row["Amount"] <= 1000:
        return "Approved"
    else:
        return "Pending"

df["Auto-Status"] = df.apply(auto_approve, axis=1)

# ------------------ AI Prediction of Completion Time ------------------
# Simulate historical completion times for prediction
historical_days = [2,3,2,5,4,6,3,2,4,5]
historical_amounts = [500,800,400,1500,1000,2000,600,700,1200,900]
X_hist = np.array(historical_amounts).reshape(-1,1)
y_hist = np.array(historical_days)
model = LinearRegression()
model.fit(X_hist, y_hist)

# Predict completion time for new tasks
df["Predicted Completion (Days)"] = df["Amount"].apply(lambda x: max(1,int(model.predict([[x]])[0])))

# ------------------ Anomaly Detection ------------------
iso = IsolationForest(contamination=0.2, random_state=42)
df['Anomaly'] = iso.fit_predict(df[['Amount', 'Predicted Completion (Days)']])
anomalies = df[df['Anomaly'] == -1]

# ------------------ KPIs ------------------
st.markdown("### Dashboard KPIs")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Tasks", len(df))
with col2:
    st.metric("Approved Tasks", len(df[df["Auto-Status"]=="Approved"]))
with col3:
    st.metric("Pending Tasks", len(df[df["Auto-Status"]=="Pending"]))
with col4:
    st.metric("Anomalous Tasks", len(anomalies))

# ------------------ Recommendations ------------------
st.markdown("### AI Recommendations")
for idx, row in df.iterrows():
    if row["Auto-Status"] == "Approved":
        st.success(f"Task {row['Task ID']} ({row['Type']}) can be auto-approved.")
    elif row["Anomaly"] == -1:
        st.warning(f"Task {row['Task ID']} ({row['Type']}) is unusual or high-risk. Requires manual review.")
    else:
        st.info(f"Task {row['Task ID']} ({row['Type']}) is pending. Assign to human approver.")

# ------------------ Graph: Task Amount vs Completion ------------------
st.markdown("### Task Visualization")
plt.figure(figsize=(8,4))
plt.scatter(df["Task ID"], df["Amount"], c=np.where(df["Anomaly"]==-1,'red','blue'), s=100)
for i, txt in enumerate(df["Predicted Completion (Days)"]):
    plt.text(df["Task ID"][i]+0.05, df["Amount"][i]+10, f"{txt}d")
plt.xlabel("Task ID")
plt.ylabel("Amount ($)")
plt.title("Tasks: Amount vs Predicted Completion Time")
plt.grid(True)
plt.show()
st.pyplot(plt)

# ------------------ Show Anomalies ------------------
if not anomalies.empty:
    st.markdown("### Detected Anomalies (High-risk tasks)")
    st.table(anomalies[["Task ID","Type","Priority","Amount","Predicted Completion (Days)"]])