"""
07_dashboard.py
CSAO RAIL - Streamlit dashboard summarizing model performance, business
impact, and A/B test results.

Run with:  streamlit run 07_dashboard.py
"""

import json
import os

import pandas as pd
import streamlit as st

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")

st.set_page_config(page_title="CSAO Rail Dashboard", layout="wide")
st.title("CSAO RAIL — Cart Super Add-On Recommendation System")
st.caption("Zomathon | Problem Statement 2 | Neural Ninjas")


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


col1, col2, col3, col4 = st.columns(4)

training_results = load_json(os.path.join(MODEL_DIR, "training_results.json"))
gen_summary = load_json(os.path.join(DATA_DIR, "generation_summary.json"))
ab_results = load_json(os.path.join(REPORT_DIR, "ab_test_results.json"))

if training_results:
    col1.metric("AUC-ROC", training_results["lightgbm"]["auc_roc"])
    col2.metric("Precision@5", training_results["lightgbm"]["precision_at_5"])
    col3.metric("F1-Score", training_results["lightgbm"]["f1_score"])
if gen_summary:
    col4.metric("CSAO Events", gen_summary["csao_events"])

st.divider()

tab1, tab2, tab3 = st.tabs(["Model Performance", "Business Impact (A/B)", "Data Overview"])

with tab1:
    st.subheader("Model Metrics")
    if training_results:
        st.json(training_results["lightgbm"])
        st.subheader("Ensemble Comparison")
        st.json(training_results["ensemble"])
    else:
        st.info("Run 04_model_training.py first to populate this tab.")

with tab2:
    st.subheader("A/B Test Results")
    if ab_results:
        st.write(f"**Decision:** {ab_results['decision']}")
        for name, metric in ab_results["metrics"].items():
            st.write(f"**{name.replace('_', ' ').title()}**")
            st.json(metric)
    else:
        st.info("Run 06_ab_testing_framework.py first to populate this tab.")

with tab3:
    st.subheader("Synthetic Dataset Summary")
    if gen_summary:
        st.json(gen_summary)
    users_path = os.path.join(DATA_DIR, "users.csv")
    if os.path.exists(users_path):
        st.write("Sample users")
        st.dataframe(pd.read_csv(users_path).head(20))
