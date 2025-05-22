import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(layout="wide")

st.title("Return Reason Splitter & Visualizer")

# Rule-based mapping from TrainingData
@st.cache_data
def load_training_data():
    df_train = pd.read_csv("TrainingData.csv")
    df_train['detailed_pv_sub_reasons'] = df_train['detailed_pv_sub_reasons'].str.strip().str.upper()
    df_train['Conclusion'] = df_train['Conclusion'].str.strip().str.upper()
    return dict(zip(df_train['detailed_pv_sub_reasons'], df_train['Conclusion']))

# File uploader
uploaded_file = st.file_uploader("Upload dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Split rows with multiple detailed_pv_sub_reasons
    df['detailed_pv_sub_reasons'] = df['detailed_pv_sub_reasons'].fillna("")
    df['detailed_pv_sub_reasons'] = df['detailed_pv_sub_reasons'].astype(str)
    df_expanded = df.drop('detailed_pv_sub_reasons', axis=1).join(
        df['detailed_pv_sub_reasons'].str.split(',', expand=True).stack().reset_index(level=1, drop=True).rename('detailed_pv_sub_reasons')
    )

    # Clean reasons
    df_expanded['detailed_pv_sub_reasons'] = df_expanded['detailed_pv_sub_reasons'].str.strip().str.upper()

    # Load mapping and apply conclusion
    mapping = load_training_data()
    df_expanded['Conclusion'] = df_expanded['detailed_pv_sub_reasons'].map(mapping)
    df_expanded['Conclusion'] = df_expanded.apply(
        lambda row: row['detailed_pv_sub_reasons'] if pd.isna(row['Conclusion']) else row['Conclusion'], axis=1
    )

    # Remove NaN and no_issue values from visualization
    df_filtered = df_expanded.copy()
    df_filtered = df_filtered.dropna(subset=['detailed_pv_sub_reasons'])
    df_filtered = df_filtered[~df_filtered['detailed_pv_sub_reasons'].str.lower().isin([
        'no_issue', 'no issue', 'noissue', 'no_issues', 'no issues', 'no-issue', 'nan', ''])]

    # Get top 10 reasons and top 20 verticals for heatmap
    top_conclusions = df_filtered['Conclusion'].value_counts().nlargest(10).index
    top_verticals = df_filtered['product_detail_cms_vertical'].value_counts().nlargest(20).index

    df_viz = df_filtered[
        df_filtered['Conclusion'].isin(top_conclusions) & 
        df_filtered['product_detail_cms_vertical'].isin(top_verticals)
    ]

    heatmap_data = pd.crosstab(df_viz['product_detail_cms_vertical'], df_viz['Conclusion'])

    # Plotting heatmap
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu", linewidths=0.5, linecolor='gray', ax=ax)
    plt.title("Top 10 Return Reasons by Top 20 Verticals", fontsize=18)
    plt.xlabel("Conclusion")
    plt.ylabel("Vertical")
    st.pyplot(fig)

    # Show transformed data (optional toggle)
    if st.checkbox("Show Transformed Data"):
        st.dataframe(df_expanded)

    # Download link for transformed CSV
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(df_expanded)
    st.download_button(
        label="Download Transformed Data as CSV",
        data=csv,
        file_name='transformed_dataset.csv',
        mime='text/csv',
    )

