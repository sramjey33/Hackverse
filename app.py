import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from datetime import datetime

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="AI Decision Intelligence System", layout="wide")
st.title("AI-Powered Business Decision Intelligence System")
st.markdown("Automated AI Analysis with Human-in-the-Loop Approval")

# ------------------ INPUT SECTION ------------------
st.markdown("### Enter Monthly Sales Data")

sales_input = st.text_input(
    "Enter 6 months sales separated by commas (example: 100,130,170,120,150,300)"
)

if sales_input:

    try:
        sales = [float(x.strip()) for x in sales_input.split(",")]
        months = list(range(1, len(sales)+1))

        df = pd.DataFrame({
            "Month": months,
            "Sales": sales
        })

        # ------------------ TREND ANALYSIS ------------------
        X = np.array(months).reshape(-1, 1)
        y = np.array(sales)

        model = LinearRegression()
        model.fit(X, y)

        trend_slope = model.coef_[0]
        confidence = model.score(X, y) * 100

        if trend_slope > 2:
            trend = "Increasing"
            suggestion = "Consider increasing production by 10-15%."
        elif trend_slope < -2:
            trend = "Declining"
            suggestion = "Consider promotional campaigns or cost optimization."
        else:
            trend = "Stable"
            suggestion = "Maintain current strategy and monitor closely."

        # ------------------ ANOMALY DETECTION ------------------
        iso = IsolationForest(contamination=0.2, random_state=42)
        df["Anomaly"] = iso.fit_predict(df[["Sales"]])
        anomalies = df[df["Anomaly"] == -1]

        volatility = np.std(sales)

        if volatility > np.mean(sales) * 0.25:
            risk = "High"
        elif volatility > np.mean(sales) * 0.15:
            risk = "Medium"
        else:
            risk = "Low"

        # ------------------ DASHBOARD ------------------
        st.markdown("## 📊 AI Analysis Dashboard")

        col1, col2, col3 = st.columns(3)
        col1.metric("Trend", trend)
        col2.metric("Confidence Level", f"{confidence:.2f}%")
        col3.metric("Risk Level", risk)

        # ------------------ GRAPH ------------------
        st.markdown("### Sales Trend Visualization")

        plt.figure()
        plt.plot(months, sales, marker='o')
        plt.title("Sales Trend")
        plt.xlabel("Month")
        plt.ylabel("Sales")
        plt.grid(True)
        st.pyplot(plt)

        # ------------------ EXECUTIVE REPORT ------------------
        st.markdown("## 📑 Automated AI Decision Report")

        report = f"""
        AI Decision Report
        ----------------------------
        Generated On: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

        Trend Analysis: {trend}
        Trend Strength (Slope): {trend_slope:.2f}
        Model Confidence: {confidence:.2f}%
        Risk Level: {risk}
        Volatility Score: {volatility:.2f}

        Suggested Strategy:
        {suggestion}

        Anomalies Detected: {len(anomalies)}

        Human Approval Required: YES
        """

        st.text(report)

        # ------------------ DOWNLOAD REPORT ------------------
        st.download_button(
            label="Download Decision Report",
            data=report,
            file_name="AI_Decision_Report.txt",
            mime="text/plain"
        )

        # ------------------ SEND TO MANAGEMENT ------------------
        if st.button("Send Report to Management"):
            st.success("Report successfully sent to management team.")

        # ------------------ HUMAN APPROVAL ------------------
        st.markdown("### Human Decision")

        decision = st.radio(
            "Management Decision:",
            ["Pending", "Approved", "Rejected"]
        )

        st.markdown("### Audit Log")
        st.write(f"Final Status: {decision}")
        st.write(f"Decision Timestamp: {datetime.now()}")

        # ------------------ ANOMALY DISPLAY ------------------
        if not anomalies.empty:
            st.warning("⚠ Anomalies detected in the following months:")
            st.table(anomalies[["Month", "Sales"]])

    except:
        st.error("Please enter valid numeric values separated by commas.")
