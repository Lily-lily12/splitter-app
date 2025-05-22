import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🔍 PV Sub Reason Splitter + Conclusion Predictor")

# Train model from TrainingData.csv
@st.cache_resource
def train_model():
    df = pd.read_csv("TrainingData.csv")
    X = df["detailed_pv_sub_reasons"].astype(str)
    y = df["Conclusion"]
    model = make_pipeline(TfidfVectorizer(), LogisticRegression())
    model.fit(X, y)
    return model

model = train_model()

# Upload interface
uploaded_file = st.file_uploader("📤 Upload your dataset CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 🔎 Original Data", df.head())

    # Step 1: Split multi-value rows
    df["detailed_pv_sub_reasons"] = df["detailed_pv_sub_reasons"].astype(str)
    df_split = df.assign(
        detailed_pv_sub_reasons=df["detailed_pv_sub_reasons"].str.split(",")
    ).explode("detailed_pv_sub_reasons").reset_index(drop=True)

    df_split["detailed_pv_sub_reasons"] = df_split["detailed_pv_sub_reasons"].str.strip()

    # Step 2: Predict Conclusion
    def predict_or_fallback(value):
        if pd.isna(value) or value.strip().lower() in ["", "no_issue", "no_issues", "no issue"]:
            return ""
        try:
            return model.predict([value])[0]
        except:
            return value

    df_split["Conclusion"] = df_split["detailed_pv_sub_reasons"].apply(predict_or_fallback)

    st.write("### ✅ Transformed & Predicted Data", df_split.head())

    # Step 3: Download link
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(df_split)
    st.download_button(
        "📥 Download Transformed CSV",
        csv,
        "transformed_data.csv",
        "text/csv"
    )

    # Step 4: Heatmap using business_unit and Conclusion
    if "business_unit" in df_split.columns:
        st.write("### 🔥 Heatmap of Conclusions by Business Unit")

        heatmap_input = df_split[
            (df_split["Conclusion"] != "") &
            (df_split["detailed_pv_sub_reasons"].str.lower().str.strip() != "no_issue")
        ]

        heatmap_data = pd.crosstab(
            heatmap_input["business_unit"],
            heatmap_input["Conclusion"]
        )

        if not heatmap_data.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt="d", linewidths=0.5, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("No valid data available for heatmap after filtering.")
    else:
        st.warning("⚠️ 'business_unit' column not found for heatmap.")


