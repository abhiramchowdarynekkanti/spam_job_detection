
import streamlit as st
import pandas as pd
import numpy as np

import streamlit.components.v1 as components
from pathlib import Path
from utils.preprocessing import preprocess_dataframe
st.write("✅ App has started running.")

st.set_option('deprecation.showPyplotGlobalUse', False)
import pickle, shap, joblib
import matplotlib.pyplot as plt
ARTIFACT_PATH = Path(__file__).parent / "model"
vectorizer = pickle.load(open(ARTIFACT_PATH / "vectorizer.pkl", "rb"))
model = pickle.load(open(ARTIFACT_PATH / "classifier.pkl", "rb"))

@st.cache_resource(show_spinner=False)
def get_explainer():
    background = vectorizer.transform(["sample text"])
    return shap.LinearExplainer(model, background, feature_perturbation="interventional")

explainer = get_explainer()

def st_shap(plot, height=400):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height, scrolling=True)

st.set_page_config(page_title="Fraudulent Job Detector", layout="wide")
st.title("🕵️‍♂️ Fraudulent Job Post Detection")

tab_upload, tab_single = st.tabs(["📁 Upload CSV", "📝 Single Entry"])

with tab_upload:
    csv_file = st.file_uploader("Upload CSV file", type=["csv"])
    if csv_file:
        with st.spinner("🔄 Please wait, processing your file..."):
            df_raw = pd.read_csv(csv_file)
            df_proc = preprocess_dataframe(df_raw)
            X_full = vectorizer.transform(df_proc["text"])
            preds = model.predict(X_full)
            decision = model.decision_function(X_full)
            prob = 1 / (1 + np.exp(-decision))
            out = df_raw.copy()
            out["fraud_prob"] = prob
            out["prediction"] = preds

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📋 Prediction Table")
            st.dataframe(out[["job_id", "title", "prediction", "fraud_prob"]])

            st.markdown("### 🔥 Top 10 Suspicious Job Listings")
            fake_top = out.sort_values("fraud_prob", ascending=False).head(10)
            st.dataframe(fake_top[["job_id", "title", "fraud_prob"]])

        with col2:
            st.subheader("📊 Histogram of Fraud Probabilities")
            fig, ax = plt.subplots()
            ax.hist(out["fraud_prob"], bins=30, color='orange', edgecolor='black')
            ax.set_xlabel("Fraud Probability")
            ax.set_ylabel("Count")
            st.pyplot(fig)

            st.subheader("🎯 Prediction Breakdown")
            pie = out["prediction"].value_counts()
            fig2, ax2 = plt.subplots()
            ax2.pie(pie, labels=["Real", "Fake"], autopct="%1.1f%%", startangle=90)
            ax2.axis("equal")
            st.pyplot(fig2)

        st.subheader("📌 SHAP Plots (Model Explanation)")
        shap_vals = explainer.shap_values(X_full)

        fig1 = plt.figure()
        shap.summary_plot(shap_vals, X_full, feature_names=vectorizer.get_feature_names_out(), show=False)

        fig2 = plt.figure()
        shap.summary_plot(shap_vals, X_full, feature_names=vectorizer.get_feature_names_out(), plot_type="bar", show=False)

        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        top_idx = np.argsort(mean_abs_shap)[-20:]
        feat_names = vectorizer.get_feature_names_out()[top_idx]
        feat_vals = mean_abs_shap[top_idx]
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        ax3.barh(feat_names, feat_vals, color='green')
        ax3.set_title("Top Features by Mean SHAP Value")

        col_shap1, col_shap2, col_shap3 = st.columns(3)
        with col_shap1:
            st.markdown("#### SHAP Summary")
            st.pyplot(fig1)
        with col_shap2:
            st.markdown("#### SHAP Bar Plot")
            st.pyplot(fig2)
        with col_shap3:
            st.markdown("#### Top SHAP Features")
            st.pyplot(fig3)

with tab_single:
    form = st.form("input_form")
    inputs = {}
    for col in [
        "title", "location", "department", "company_profile",
        "description", "requirements", "benefits",
        "employment_type", "required_experience", "required_education",
        "industry", "function"]:
        inputs[col] = form.text_area(f"{col.capitalize()}", height=60)
    col_tele, col_logo = form.columns(2)
    inputs["telecommuting"] = col_tele.selectbox("Telecommuting?", [0, 1])
    inputs["has_company_logo"] = col_logo.selectbox("Has Company Logo?", [0, 1])

    submitted = form.form_submit_button("Predict")
    if submitted:
        df_one = pd.DataFrame([inputs])
        df_proc = preprocess_dataframe(df_one)
        X_text = vectorizer.transform(df_proc["text"])
        X_full = X_text

        pred = model.predict(X_full)[0]
        decision = model.decision_function(X_full)[0]
        prob = 1 / (1 + np.exp(-decision))
        label = "🚨 Fake" if pred == 1 else "✅ Real"
        st.markdown(f"## **{label}**  \nProbability: **{prob:.2%}**")

        shap_vals = explainer.shap_values(X_text)
        abs_shap = np.abs(shap_vals).flatten()
        idxs = np.argsort(abs_shap)[-20:]
        feat_names = vectorizer.get_feature_names_out()[idxs]
        shap_imp = abs_shap[idxs]
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.barh(feat_names, shap_imp, color="purple")
        ax3.set_title("Top Terms Influencing This Prediction")
        st.pyplot(fig3)

        fig_w = shap.plots._waterfall.waterfall_legacy(explainer(X_text)[0], show=False)
        st.pyplot(fig_w)

        force_plot = shap.plots.force(explainer.expected_value, shap_vals[0], matplotlib=False)
        st_shap(force_plot)
