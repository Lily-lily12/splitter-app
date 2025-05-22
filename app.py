import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import base64

# ------------------ MODEL TRAINING FUNCTION ------------------ #
def train_model():
    df = pd.read_csv("TrainingData.csv")
    df = df.dropna(subset=["detailed_pv_sub_reasons", "Conclusion"])

    # Remove rows with "no_issue"-like values
    ignore_values = ['no_issue', 'no_Issue', 'NO_ISSUE']
    df = df[~df['detailed_pv_sub_reasons'].isin(ignore_values)]

    X = df["detailed_pv_sub_reasons"].astype(str)
    y = df["Conclusion"].astype(str)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X, y_encoded)

    joblib.dump(pipeline, "model.pkl")
    joblib.dump(encoder, "encoder.pkl")

    return pipeline, encoder

# ------------------ PREDICTION FUNCTION ------------------ #
def predict_conclusion(value, model, encoder):
    try:
        if pd.isna(value) or value.lower() in ['no_issue', 'no_Issue', 'NO_ISSUE']:
            return value
        pred = model.predict([value])
        return encoder.inverse_transform(pred)[0]
    except:
        return value

# ------------------ DATA TRANSFORMATION FUNCTION ------------------ #
def split_rows(df, column):
    rows = []
    for _, row in df.iterrows():
        values = str(row[column]).split(',')
        for val in values:
            new_row = row.copy()
            new_row[column] = val.strip()
            rows.append(new_row)
    return pd.DataFrame(rows)

# ------------------ DOWNLOAD LINK ------------------ #
def get_table_download_link(df, filename="transformed_data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download Transformed Data</a>'

# ------------------ HEATMAP FUNCTION ------------------ #
def generate_heatmap(df):
    pivot = df.pivot_table(index="product_detail_cms_vertical", 
                           columns="detailed_pv_sub_reasons", 
                           aggfunc='size', fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot, cmap="YlGnBu", linewidths=0.5, annot=True, fmt='d')
    st.pyplot(fig)

# ------------------ STREAMLIT UI ------------------ #
st.set_page_config(page_title="Splitter App", layout="wide")
st.title("🔍 Splitter App with ML Conclusion Predictor")

st.markdown("Upload your dataset to split multi-reason rows, predict conclusions, and generate a visual heatmap.")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Load or train model
if not os.path.exists("model.pkl") or not os.path.exists("encoder.pkl"):
    with st.spinner("Training model..."):
        model, encoder = train_model()
else:
    model = joblib.load("model.pkl")
    encoder = joblib.load("encoder.pkl")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Split multiple values
    df_split = split_rows(df, "detailed_pv_sub_reasons")

    # Predict conclusions
    df_split["Conclusion"] = df_split["detailed_pv_sub_reasons"].apply(lambda x: predict_conclusion(x, model, encoder))

    st.success("✅ Data transformed and predictions added.")
    st.write(df_split.head())

    # Download link
    st.markdown(get_table_download_link(df_split), unsafe_allow_html=True)

    # Visualize
    st.subheader("📊 Heatmap of detailed reasons by product vertical")
    generate_heatmap(df_split)
