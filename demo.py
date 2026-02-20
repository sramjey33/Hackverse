#!/usr/bin/env python3
"""
AI-Driven Decision Support System - Demo Script
==============================================

This script helps you quickly test the AI DSS application with sample data.

Usage:
1. Make sure all dependencies are installed: pip install -r requirements.txt
2. Run this script: python demo.py
3. Or run the Streamlit app directly: streamlit run app.py

Demo Datasets Available:
- Growth Scenario: Steady upward trend
- Decline Scenario: Steady downward trend
- Volatile Scenario: High variability with anomalies
- Seasonal Pattern: Cyclical patterns

Features Demonstrated:
✅ Predictive Analytics (ARIMA + Exponential Smoothing)
✅ Prescriptive Analytics (AI Recommendations)
✅ Smart Operations Recommendations
✅ Automated Root-Cause Analysis
✅ Email Automation (configure in sidebar)
✅ Scenario Planning (Best/Base/Worst cases)
✅ Real-time Alerts & Notifications
✅ Human-in-the-Loop Decision Workflow

Hackathon Problem Statement Addressed:
- AI-Driven Decision Support Systems
- Predictive and prescriptive analytics
- Smart recommendations for operations
- Automated root-cause analysis

For the hackathon demo:
1. Load one of the demo datasets
2. Configure email settings (optional)
3. Run analysis and show results
4. Demonstrate email automation
5. Show scenario planning capabilities

Author: Hackathon 2026 Team
"""

import sys
import subprocess

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'matplotlib',
        'seaborn', 'scikit-learn', 'statsmodels'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n📦 Install with: pip install -r requirements.txt")
        return False

    print("✅ All dependencies are installed!")
    return True

def run_demo():
    """Run the Streamlit application"""
    print("🚀 Starting AI-Driven Decision Support System...")
    print("📱 App will be available at: http://localhost:8501")
    print("\n💡 Demo Tips:")
    print("   1. Try the demo datasets in the 'Hackathon Demo & Testing' section")
    print("   2. Configure email settings in the sidebar for automation demo")
    print("   3. Explore different analysis tabs and scenario planning")
    print("   4. Test the email templates for automated communication")
    print("\n🎯 Press Ctrl+C to stop the application")

    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped. Thanks for testing!")
    except Exception as e:
        print(f"❌ Error running application: {e}")
        print("💡 Make sure streamlit is installed: pip install streamlit")

if __name__ == "__main__":
    print("🤖 AI-Driven Decision Support System - Hackathon Demo")
    print("=" * 55)

    if check_dependencies():
        print("\n🎯 Ready to launch the application!")
        input("Press Enter to start the demo... ")
        run_demo()
    else:
        print("\n❌ Please install missing dependencies first.")
        sys.exit(1)