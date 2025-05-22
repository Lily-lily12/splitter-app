import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="Return Reason Splitter", layout="wide")
st.title("📦 Return Reason Splitter")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()  # Remove extra whitespace from column names

    # Drop rows with missing essential data
    df = df.dropna(subset=['detailed_pv_sub_reasons', 'conclusion'])

    # Normalize column values
    df['detailed_pv_sub_reasons'] = df['detailed_pv_sub_reasons'].str.lower().str.strip()
    df['conclusion'] = df['conclusion'].str.lower().str.strip()
    df['product_detail_cms_vertical'] = df['product_detail_cms_vertical'].str.lower().str.strip()

    # Remove 'no_issue' from detailed_pv_sub_reasons
    df = df[~df['detailed_pv_sub_reasons'].str.contains('no_issue', na=False)]

    # Split multiple reasons into separate rows
    df = df.assign(detailed_pv_sub_reasons=df['detailed_pv_sub_reasons'].str.split(","))
    df = df.explode('detailed_pv_sub_reasons')
    df['detailed_pv_sub_reasons'] = df['detailed_pv_sub_reasons'].str.strip()

    # Remove rows with 'no_issue' again if any slipped through
    df = df[~df['detailed_pv_sub_reasons'].str.contains('no_issue', na=False)]

    # Filter to top N reasons and top M verticals
    top_n_reasons = 10
    top_m_verticals = 20
    top_reasons = df['detailed_pv_sub_reasons'].value_counts().nlargest(top_n_reasons).index
    top_verticals = df['product_detail_cms_vertical'].value_counts().nlargest(top_m_verticals).index

    df_filtered = df[df['detailed_pv_sub_reasons'].isin(top_reasons)]
    df_filtered = df_filtered[df_filtered['product_detail_cms_vertical'].isin(top_verticals)]

    # Create pivot table
    heatmap_data = pd.pivot_table(
        df_filtered,
        index='product_detail_cms_vertical',
        columns='detailed_pv_sub_reasons',
        values='conclusion',
        aggfunc='count',
        fill_value=0
    )

    # Heatmap visualization
    st.markdown("### 🔥 Heatmap: Top Return Reasons by Vertical")
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu", linewidths=0.5, ax=ax)
    ax.set_xlabel("Conclusion")
    ax.set_ylabel("Vertical")
    st.pyplot(fig)

    # Optionally show transformed data
    if st.checkbox("Show Transformed Data"):
        st.dataframe(df_filtered)

    # Download processed CSV
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button("Download Transformed Data", csv, "transformed_data.csv", "text/csv")

