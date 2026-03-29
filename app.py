import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from Inference.freight_predict import predict_freight_cost
from Inference.invoice_flagging_prediction import predict_invoice_flag

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    page_icon="📦",
    layout="wide"
)

# ---------------------------------------------------------
# Header Section
# ---------------------------------------------------------
st.markdown("""
# 📦 Vendor Invoice Intelligence Portal
### AI-Driven Freight Cost Prediction & Invoice Risk Flagging

This internal analytics portal leverages machine learning to:
- **Forecast freight costs accurately**
- **Detect invoice anomalies**
""")

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.title("🔍 Model Selection")

selected_model = st.sidebar.radio(
    "Choose Prediction Module",
    [
        "Freight Cost Prediction",
        "Invoice Manual Approval Flag"
    ]
)

st.sidebar.markdown("""
---
**Business Impact**
- 📉 Improved cost forecasting  
- 🛡️ Reduced invoice fraud & anomalies  
- ⚙️ Faster finance operations  
""")

# ---------------------------------------------------------
# Freight Cost Prediction
# ---------------------------------------------------------
if selected_model == "Freight Cost Prediction":
    st.subheader("🚛 Freight Cost Prediction")

    with st.form("freight_form"):
        col1, col2 = st.columns(2)

        with col1:
            quantity = st.number_input("📦 Quantity", min_value=1, value=1200)

        with col2:
            dollars = st.number_input("💰 Invoice Dollars", min_value=1.0, value=18500.0)

        submit_freight = st.form_submit_button("Predict Freight Cost")

    if submit_freight:
        input_data = {
            "Dollars": [dollars]
        }

        prediction_df = predict_freight_cost(input_data)
        predicted_cost = prediction_df["Predicted_Freight"].iloc[0]

        st.success(f"### Predicted Freight Cost: ${predicted_cost:,.2f}")

        st.info("""
        **Note:** Prediction is based on historical patterns.  
        Actual costs may vary due to real-world factors like carrier charges.
        """)

# ---------------------------------------------------------
# Invoice Manual Approval Flag
# ---------------------------------------------------------
if selected_model == "Invoice Manual Approval Flag":
    st.subheader("🚩 Invoice Manual Approval Flagging")

    with st.form("flag_form"):
        col1, col2 = st.columns(2)

        with col1:
            inv_qty = st.number_input("📦 Invoice Quantity", min_value=1, value=100)
            inv_dollars = st.number_input("💰 Invoice Dollars", min_value=1.0, value=5000.0)
            freight = st.number_input("🚛 Freight Cost", min_value=0.0, value=250.0)

        with col2:
            item_qty = st.number_input("🔢 Total Item Quantity", min_value=1, value=100)
            item_dollars = st.number_input("💵 Total Item Dollars", min_value=1.0, value=5000.0)

        submit_flag = st.form_submit_button("Evaluate Invoice Risk")

    if submit_flag:
        input_data = {
            "invoice_quantity": [inv_qty],
            "invoice_dollars": [inv_dollars],
            "Freight": [freight],
            "total_item_quantity": [item_qty],
            "total_item_dollars": [item_dollars]
        }

        prediction_df = predict_invoice_flag(input_data)
        is_flagged = prediction_df["Predicted_Flag"].iloc[0]

        if is_flagged == 1:
            st.error("🚨 Invoice requires MANUAL APPROVAL")
            st.warning("Anomalies detected in invoice data.")
        else:
            st.success("✅ Invoice is SAFE for Auto-Approval")
            st.balloons()