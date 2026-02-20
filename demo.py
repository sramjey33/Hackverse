import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="IPA Dashboard", layout="wide")

st.title("Intelligent Process Automation Dashboard")

st.markdown("""
AI-powered workflow automation system that:
- Automates routine approvals
- Predicts completion timelines
- Detects anomalies
""")

# ------------------ INPUT ------------------
st.header("Task Entry")

col1, col2, col3 = st.columns(3)

with col1:
    t1_priority = st.selectbox("Task 1 Priority", ["Low", "Medium", "High"])
    t1_amount = st.number_input("Task 1 Amount", min_value=0, value=500)

with col2:
    t2_priority = st.selectbox("Task 2 Priority", ["Low", "Medium", "High"])
    t2_amount = st.number_input("Task 2 Amount", min_value=0, value=1200)

with col3:
    t3_priority = st.selectbox("Task 3 Priority", ["Low", "Medium", "High"])
    t3_amount = st.number_input("Task 3 Amount", min_value=0, value=800)

data = [
    {"Task ID": 1, "Priority": t1_priority, "Amount": t1_amount},
    {"Task ID": 2, "Priority": t2_priority, "Amount": t2_amount},
    {"Task ID": 3, "Priority": t3_priority, "Amount": t3_amount},
]

df = pd.DataFrame(data)

# ------------------ AUTO APPROVAL ------------------
def auto_rule(row):
    if row["Priority"] == "Low" or row["Amount"] <= 1000:
        return "Approved"
    return "Review"

df["Status"] = df.apply(auto_rule, axis=1)

# ------------------ PREDICTION MODEL ------------------
hist_amt = np.array([500,800,400,1500,1000,2000,600,700,1200,900]).reshape(-1,1)
hist_days = np.array([2,3,2,5,4,6,3,2,4,5])

model = LinearRegression()
model.fit(hist_amt, hist_days)

df["Predicted Days"] = df["Amount"].apply(
    lambda x: max(1, int(model.predict([[x]])[0]))
)

# ------------------ ANOMALY DETECTION ------------------
iso = IsolationForest(contamination=0.2, random_state=42)
df["Risk Flag"] = iso.fit_predict(df[["Amount", "Predicted Days"]])

anomalies = df[df["Risk Flag"] == -1]

# ------------------ KPIs ------------------
st.header("Key Metrics")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Tasks", len(df))
c2.metric("Auto Approved", len(df[df["Status"] == "Approved"]))
c3.metric("Needs Review", len(df[df["Status"] == "Review"]))
c4.metric("High Risk", len(anomalies))

# ------------------ RECOMMENDATIONS ------------------
st.header("AI Recommendations")

for _, row in df.iterrows():
    if row["Status"] == "Approved":
        st.success(f"Task {row['Task ID']} can be processed automatically.")
    elif row["Risk Flag"] == -1:
        st.warning(f"Task {row['Task ID']} appears high-risk. Manual validation required.")
    else:
        st.info(f"Task {row['Task ID']} requires approval.")

# ------------------ VISUAL ------------------
st.header("Task Analysis")

fig, ax = plt.subplots()

colors = ["red" if x == -1 else "blue" for x in df["Risk Flag"]]

ax.scatter(df["Task ID"], df["Amount"], c=colors)

for i in range(len(df)):
    ax.text(
        df["Task ID"][i],
        df["Amount"][i],
        f"{df['Predicted Days'][i]}d"
    )

ax.set_xlabel("Task ID")
ax.set_ylabel("Amount")
ax.grid(True)

st.pyplot(fig)

if not anomalies.empty:
    st.header("High Risk Tasks")
    st.dataframe(anomalies)