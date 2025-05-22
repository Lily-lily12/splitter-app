import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

# --- Function to split the multi-reason rows ---
def split_rows(df, col):
    df[col] = df[col].astype(str)
    df[col] = df[col].str.replace(' ', '')
    df[col] = df[col].str.upper()
    df = df.assign(**{col: df[col].str.split(',')})
    df = df.explode(col).reset_index(drop=True)
    return df

# --- Train model from TrainingData.csv ---
@st.cache_resource
def train_model():
    train_df = pd.read_csv("TrainingData.csv")
    X = train_df["detailed_pv_sub_reasons"].astype(str)
    y = train_df["Conclusion"]
    
    model = make_pipeline(TfidfVectorizer(), LogisticRegression())
    model.fit(X, y)
    return model

# --- Predict conclusion using model ---
def predict_conclusion(model, df, col):
    try:
        df["Conclusion"] = model.predict(df[col])
    except:
        df["Conclusion"] = df[col]
    return df

# --- Main Streamlit app ---
st.title("🔍 PV Sub-Reason Splitter + Predictor")

uploaded_file = st.file_uploader("📂 Upload your dataset (CSV file)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    
    if "detailed_pv_sub_reasons" not in data.columns or "product_detail_cms_vertical" not in data.columns:
        st.error("CSV must contain 'detailed_pv_sub_reasons' and 'product_detail_cms_vertical' columns.")
    else:
        st.subheader("📊 Raw Data Preview")
        st.dataframe(data.head())

        # Process data
        model = train_model()
        processed = split_rows(data, "detailed_pv_sub_reasons")
        processed = predict_conclusion(model, processed, "detailed_pv_sub_reasons")

        st.subheader("✅ Transformed Data Preview")
        st.dataframe(processed.head())

        # --- Download transformed CSV ---
        buffer = BytesIO()
        processed.to_csv(buffer, index=False)
        buffer.seek(0)
        st.download_button("⬇️ Download Transformed CSV", buffer, "processed_data.csv", "text/csv")

        # --- Heatmap-like visualization ---
        st.subheader("🔥 Issue Frequency by Vertical")

        heatmap_data = processed.groupby(["product_detail_cms_vertical", "detailed_pv_sub_reasons"]).size().unstack(fill_value=0)

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt="d", ax=ax)
        st.pyplot(fig)
