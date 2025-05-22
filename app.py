import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import base64
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Return Reason Splitter", layout="wide")

# Load rule-based training data (already cleaned and augmented)
@st.cache_data
def load_training_data():
    return pd.read_csv("TrainingData_AUGMENTED.csv")

# Rule-based mapper for Conclusion from training data
def create_rule_based_mapper(training_df):
    mapper = dict()
    for _, row in training_df.iterrows():
        reason = str(row['detailed_pv_sub_reasons']).strip().upper()
        conclusion = str(row['Conclusion']).strip()
        mapper[reason] = conclusion
    return mapper

# Split rows with comma-separated values
def split_rows(df, column):
    rows = []
    for _, row in df.iterrows():
        values = str(row[column]).split(',')
        for val in values:
            new_row = row.copy()
            new_row[column] = val.strip()
            rows.append(new_row)
    return pd.DataFrame(rows)

# Rule-based prediction of Conclusion
def apply_rule_based_prediction(df, mapper):
    def map_reason(reason):
        reason_key = str(reason).strip().upper()
        return mapper.get(reason_key, reason)

    df['Conclusion'] = df['detailed_pv_sub_reasons'].apply(map_reason)
    return df

# Remove invalid entries
def clean_data_for_heatmap(df):
    df = df.dropna(subset=['detailed_pv_sub_reasons'])
    df = df[~df['detailed_pv_sub_reasons'].str.upper().isin(['NO_ISSUE', 'NOISSUE', 'NO ISSUE'])]
    return df

# Heatmap function
def plot_heatmap(df):
    pivot = pd.pivot_table(
        df,
        index='business_unit',
        columns='detailed_pv_sub_reasons',
        aggfunc='size',
        fill_value=0
    )
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="Blues", linewidths=0.5, ax=ax)
    st.pyplot(fig)

# Convert DataFrame to download link
def get_table_download_link(df):
    towrite = BytesIO()
    df.to_csv(towrite, index=False)
    towrite.seek(0)
    b64 = base64.b64encode(towrite.read()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="transformed_dataset.csv">Download Transformed CSV File</a>'
    return href

# UI
st.title("📦 Return Reason Splitter & Classifier")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    if 'detailed_pv_sub_reasons' not in input_df.columns:
        st.error("The uploaded file must contain the column 'detailed_pv_sub_reasons'")
    else:
        training_data = load_training_data()
        mapper = create_rule_based_mapper(training_data)

        # Step 1: Split rows
        df_split = split_rows(input_df, 'detailed_pv_sub_reasons')

        # Step 2: Predict Conclusion
        df_predicted = apply_rule_based_prediction(df_split, mapper)

        # Step 3: Show data preview
        st.subheader("🔍 Transformed Data Preview")
        st.dataframe(df_predicted.head(50))

        # Step 4: Download link
        st.markdown(get_table_download_link(df_predicted), unsafe_allow_html=True)

        # Step 5: Plot heatmap
        if 'business_unit' in df_predicted.columns:
            st.subheader("📊 Heatmap: Reasons by Business Unit")
            df_cleaned = clean_data_for_heatmap(df_predicted)
            plot_heatmap(df_cleaned)
        else:
            st.warning("Column 'business_unit' not found in the dataset. Heatmap skipped.")
else:
    st.info("Please upload a CSV file to get started.")


