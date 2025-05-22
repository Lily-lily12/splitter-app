import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Return Reason Splitter & Visualizer")

# Rule-based mapping from TrainingData
@st.cache_data
def load_training_data():
    df_train = pd.read_csv("TrainingData.csv")
    df_train['detailed_pv_sub_reasons'] = df_train['detailed_pv_sub_reasons'].str.strip().str.upper()
    df_train['Conclusion'] = df_train['Conclusion'].str.strip().str.upper()
    return dict(zip(df_train['detailed_pv_sub_reasons'], df_train['Conclusion']))

# CSS style for square metric boxes
square_style = """
<style>
.metric-square {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    width: 150px;
    height: 150px;
    border: 2px solid #4CAF50;
    border-radius: 10px;
    background-color: #E8F5E9;
    margin: 10px;
    font-family: 'Arial', sans-serif;
}
.metric-title {
    font-size: 20px;
    font-weight: 700;
    color: #2E7D32;
}
.metric-value {
    font-size: 36px;
    font-weight: 900;
    margin-top: 10px;
    color: #1B5E20;
}
</style>
"""

st.markdown(square_style, unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Upload dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'business_type' in df.columns:
        total_rto = len(df[df['business_type'].str.lower() == 'rto'])
        total_rvp = len(df[df['business_type'].str.lower() == 'rvp'])
        total = len(df)

        rto_ri = (total_rto / total) * 100 if total > 0 else 0
        rvp_ri = (total_rvp / total) * 100 if total > 0 else 0

        # Show side by side squares with RTO RI and RVP RI
        col1, col2 = st.columns(2)

        col1.markdown(f"""
        <div class="metric-square">
            <div class="metric-title">RTO RI</div>
            <div class="metric-value">{rto_ri:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

        col2.markdown(f"""
        <div class="metric-square">
            <div class="metric-title">RVP RI</div>
            <div class="metric-value">{rvp_ri:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # Rest of your processing here (split, mapping, heatmap, download, etc.)

    # Split rows with multiple detailed_pv_sub_reasons
    df['detailed_pv_sub_reasons'] = df['detailed_pv_sub_reasons'].fillna("")
    df['detailed_pv_sub_reasons'] = df['detailed_pv_sub_reasons'].astype(str)
    df_expanded = df.drop('detailed_pv_sub_reasons', axis=1).join(
        df['detailed_pv_sub_reasons'].str.split(',', expand=True).stack().reset_index(level=1, drop=True).rename('detailed_pv_sub_reasons')
    )

    df_expanded['detailed_pv_sub_reasons'] = df_expanded['detailed_pv_sub_reasons'].str.strip().str.upper()

    mapping = load_training_data()
    df_expanded['Conclusion'] = df_expanded['detailed_pv_sub_reasons'].map(mapping)
    df_expanded['Conclusion'] = df_expanded.apply(
        lambda row: row['detailed_pv_sub_reasons'] if pd.isna(row['Conclusion']) else row['Conclusion'], axis=1
    )

    df_filtered = df_expanded.copy()
    df_filtered = df_filtered.dropna(subset=['detailed_pv_sub_reasons'])
    df_filtered = df_filtered[~df_filtered['detailed_pv_sub_reasons'].str.lower().isin([
        'no_issue', 'no issue', 'noissue', 'no_issues', 'no issues', 'no-issue', 'nan', ''
    ])]

    top_conclusions = df_filtered['Conclusion'].value_counts().nlargest(10).index
    top_verticals = df_filtered['product_detail_cms_vertical'].value_counts().nlargest(20).index

    df_viz = df_filtered[
        df_filtered['Conclusion'].isin(top_conclusions) &
        df_filtered['product_detail_cms_vertical'].isin(top_verticals)
    ]

    heatmap_data = pd.crosstab(df_viz['product_detail_cms_vertical'], df_viz['Conclusion'])

    fig, ax = plt.subplots(figsize=(20, 10))
    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu", linewidths=0.5, linecolor='gray', ax=ax)
    plt.title("Top 10 Return Reasons by Top 20 Verticals", fontsize=18)
    plt.xlabel("Conclusion")
    plt.ylabel("Vertical")
    st.pyplot(fig)

    if st.checkbox("Show Transformed Data"):
        st.dataframe(df_expanded)

    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(df_expanded)
    st.download_button(
        label="Download Transformed Data as CSV",
        data=csv,
        file_name='transformed_dataset.csv',
        mime='text/csv',
    )
