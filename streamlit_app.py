"""Credit card fraud detection: graph ML live demo (Phase 2).

Run from the project root:  streamlit run streamlit_app.py
Requires the artifacts/ folder produced by the offline pipeline (see README.md).
"""
import streamlit as st

st.set_page_config(page_title="Graph Fraud Detection Demo", page_icon="🕸️", layout="wide")

pages = st.navigation([
    st.Page("app/pages/predict.py", title="Predict", icon="🔎", default=True),
    st.Page("app/pages/model_card.py", title="Model Card", icon="📋"),
    st.Page("app/pages/about.py", title="About / Methodology", icon="📖"),
])
pages.run()
