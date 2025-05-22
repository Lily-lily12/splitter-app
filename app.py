import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(layout="wide")
st.title("📊 Cleaned Heatmap: Reasons vs Conclusion")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()

    # Drop NaNs and rows with 'no_issue'
    df = df.dropna(subset=['detailed_pv_sub_reasons', 'conclusion'])
    df = df[~df['detailed_pv_sub_reasons'].str.lower().str.contains("no_issue")]

    # Normalize and split multiple sub-reasons
    df['detailed_pv_sub_reasons'] = df['detailed_pv_sub_reasons'].str.upper().str.replace(" ", "")
    df = df.assign(detailed_pv_sub_reasons=df['detailed_pv_sub_reasons'].str.split(','))
    df = df.explode('detailed_pv_sub_reasons')

    # Remove 'NO_ISSUE' again in case it's nested
    df = df[~df['detailed_pv_sub_reasons'].str.lower().str.contains("no_issue")]

    # Clean duplicates due to formatting
    df['detailed_pv_sub_reasons'] = df['detailed_pv_sub_reasons'].str.strip().str.upper()
    df['conclusion'] = df['conclusion'].str.strip().str.upper()

    # Filter top 10 sub reasons and top 20 conclusions
    top_reasons = df['detailed_pv_sub_reasons'].value_counts().nlargest(10).index
    top_conclusions = df['conclusion'].value_counts().nlargest(20).index

    df_filtered = df[df['detailed_pv_sub_reasons'].isin(top_reasons) & df['conclusion'].isin(top_conclusions)]

    # Pivot for heatmap
    heatmap_data = df_filtered.pivot_table(index='detailed_pv_sub_reasons', 
                                           columns='conclusion', 
                                           aggfunc='size', 
                                           fill_value=0)

    # Plotting
    plt.figure(figsize=(18, 8))
    sns.set(font_scale=0.9)
    ax = sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu")
    plt.xlabel("Conclusion")
    plt.ylabel("Detailed PV Sub Reasons")
    plt.title("🔍 Top Reasons vs Conclusion Heatmap")
    st.pyplot(plt.gcf())

    # Show transformed data toggle
    if st.checkbox("Show Transformed Data"):
        st.dataframe(df_filtered)

    # Download transformed dataset
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Transformed CSV",
        data=csv,
        file_name='transformed_data.csv',
        mime='text/csv',
    )

