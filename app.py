import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🔍 PV Sub Reason Splitter + Conclusion Predictor")

# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# Load training data and generate embeddings
@st.cache_resource
def prepare_embeddings():
    df = pd.read_csv("TrainingData.csv")
    df["detailed_pv_sub_reasons"] = df["detailed_pv_sub_reasons"].astype(str)
    embeddings = embedding_model.encode(df["detailed_pv_sub_reasons"].tolist(), convert_to_tensor=True)
    return df, embeddings

training_df, training_embeddings = prepare_embeddings()

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
        value = str(value).strip()
        if value == "" or value.lower() in ["no_issue", "no_issues", "no issue", "no_Issue"]:
            return ""
        try:
            val_embedding = embedding_model.encode([value], convert_to_tensor=True)
            cos_scores = cosine_similarity(val_embedding.cpu().numpy(), training_embeddings.cpu().numpy())[0]
            best_match_index = np.argmax(cos_scores)
            return training_df.iloc[best_match_index]["Conclusion"]
        except:
            return value

    df_split["Conclusion"] = df_split["detailed_pv_sub_reasons"].apply(predict_or_fallback)

    # Step 3: Remove blanks and "no issue" from visualization
    df_filtered = df_split[df_split["Conclusion"].str.strip() != ""]

    st.write("### ✅ Transformed & Predicted Data", df_split.head())

    # Step 4: Download link
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

    # Step 5: Heatmap
    if "business_unit" in df_filtered.columns:
        st.write("### 🔥 Heatmap of Conclusion by Business Unit")
        heatmap_data = pd.crosstab(
            df_filtered["business_unit"],
            df_filtered["Conclusion"]
        )

        fig, ax = plt.subplots(figsize=(15, 6))
        sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt="d", linewidths=0.5, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("⚠️ 'business_unit' column not found for heatmap.")
