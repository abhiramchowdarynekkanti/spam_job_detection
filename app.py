import pickle, shap, joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils.preprocessing import preprocess_dataframe

ARTIFACT_PATH = Path(__file__).parent / "model"
vectorizer = pickle.load(open(ARTIFACT_PATH / "vectorizer.pkl", "rb"))
model      = pickle.load(open(ARTIFACT_PATH / "classifier.pkl", "rb"))

@st.cache_resource(show_spinner=False)
def get_explainer():
    return shap.LinearExplainer(model, vectorizer.transform(["dummy"]))

explainer = get_explainer()

st.set_page_config(page_title="Fraudulent Job Detector", layout="wide")
st.title("🕵️‍♂️ Fraudulent Job Post Detection")

tab_upload, tab_single = st.tabs(["📁 Upload CSV", "📝 Single Entry"])

with tab_upload:
    st.header("Batch prediction from CSV")
    csv_file = st.file_uploader("Upload CSV file", type=["csv"])
    if csv_file:
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
            st.subheader("Prediction table")
            st.dataframe(out[["job_id","title","prediction","fraud_prob"]])
            fake_top = out.sort_values("fraud_prob", ascending=False).head(10)
            st.markdown("### 🔥 Top-10 most suspicious")
            st.dataframe(fake_top[["job_id","title","fraud_prob"]])
        with col2:
            st.subheader("Histogram of fraud probabilities")
            fig, ax = plt.subplots()
            ax.hist(out["fraud_prob"], bins=30)
            ax.set_xlabel("Fraud probability")
            st.pyplot(fig)
            st.subheader("Fake vs Real")
            pie = out["prediction"].value_counts()
            fig2, ax2 = plt.subplots()
            ax2.pie(pie, labels=["Real","Fake"], autopct="%1.1f%%")
            st.pyplot(fig2)

with tab_single:
    st.header("Manual form input")
    form = st.form("input_form")
    inputs = {}
    for col in [
        "title", "location", "department", "company_profile",
        "description", "requirements", "benefits",
        "employment_type", "required_experience", "required_education",
        "industry", "function"
    ]:
        inputs[col] = form.text_area(f"{col.capitalize()}", height=60)
    col_tele, col_logo = form.columns(2)
    inputs["telecommuting"] = col_tele.selectbox("Telecommuting?", [0,1])
    inputs["has_company_logo"] = col_logo.selectbox("Has Company Logo?", [0,1])

    submitted = form.form_submit_button("Predict")
    if submitted:
        df_one = pd.DataFrame([inputs])
        df_proc = preprocess_dataframe(df_one)
        X_text = vectorizer.transform(df_proc["text"])
        X_full = np.hstack([X_text.toarray(),
                            df_proc[["character_count","telecommuting",
                                     "has_company_logo"]].values])
        pred = model.predict(X_full)[0]
        decision = model.decision_function(X_full)[0]
        prob = 1/(1+np.exp(-decision))
        label = "🚨 Fake" if pred==1 else "✅ Real"
        st.markdown(f"## **{label}**  \nProbability: **{prob:.2%}**")
        shap_vals = explainer.shap_values(X_text)
        abs_shap = np.abs(shap_vals).flatten()
        idxs = np.argsort(abs_shap)[-20:]
        feat_names = vectorizer.get_feature_names_out()[idxs]
        shap_imp = abs_shap[idxs]
        fig3, ax3 = plt.subplots(figsize=(6,4))
        ax3.barh(feat_names, shap_imp)
        ax3.set_title("Top terms influencing the decision")
        st.pyplot(fig3)
