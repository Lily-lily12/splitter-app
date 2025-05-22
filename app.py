import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="PV Sub Reason Splitter & Visualizer", layout="wide")
st.title("📦 Return Reason Splitter & Visualizer")

st.markdown("""
Upload a dataset that contains:
- `detailed_pv_sub_reasons` (comma-separated values allowed)
- `product_detail_cms_vertical` (for visualization)
- `business_unit` (optional but used previously)

The app will:
1. Split multi-reason rows into separate rows
2. Add a rule-based `Conclusion` column from training data
3. Provide a download link for the transformed data
4. Show a heatmap of top reasons grouped by vertical
""")

# Upload main data
main_file = st.file_uploader("Upload your dataset CSV", type="csv")
training_file = st.file_uploader("Upload TrainingData.csv", type="csv")

if main_file and training_file:
    # Load main and training datasets
    df = pd.read_csv(main_file)
    training_df = pd.read_csv(training_file)

    if 'detailed_pv_sub_reasons' not in df.columns or 'product_detail_cms_vertical' not in df.columns:
        st.error("Both required columns 'detailed_pv_sub_reasons' and 'product_detail_cms_vertical' must exist in your data.")
    else:
        # --- Step 1: Split multiple reasons into rows ---
        df['detailed_pv_sub_reasons'] = df['detailed_pv_sub_reasons'].astype(str).str.split(',')
        df = df.explode('detailed_pv_sub_reasons')
        df['detailed_pv_sub_reasons'] = df['detailed_pv_sub_reasons'].str.strip().str.lower()

        # --- Step 2: Rule-based Conclusion from training ---
        training_df['detailed_pv_sub_reasons'] = training_df['detailed_pv_sub_reasons'].str.strip().str.lower()
        training_df['Conclusion'] = training_df['Conclusion'].str.strip()
        reason_to_conclusion = dict(zip(training_df['detailed_pv_sub_reasons'], training_df['Conclusion']))

        def get_conclusion(reason):
            return reason_to_conclusion.get(reason, reason)

        df['Conclusion'] = df['detailed_pv_sub_reasons'].apply(get_conclusion)

        # --- Step 3: Clean data (remove no_issue or NaN) ---
        exclude = ['no_issue', 'no issue', 'no', 'nan']
        df = df[~df['detailed_pv_sub_reasons'].isin(exclude)]
        df = df.dropna(subset=['detailed_pv_sub_reasons', 'product_detail_cms_vertical'])

        # --- Step 4: Filter Top N reasons ---
        top_n = 15
        top_reasons = df['detailed_pv_sub_reasons'].value_counts().nlargest(top_n).index
        df_filtered = df[df['detailed_pv_sub_reasons'].isin(top_reasons)]

        # --- Step 5: Download transformed data ---
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Transformed CSV",
            data=csv,
            file_name='transformed_dataset.csv',
            mime='text/csv'
        )

        # --- Step 6: Show transformed data preview ---
        st.subheader("🔍 Preview of Transformed Data")
        st.dataframe(df.head(50))

        # --- Step 7: Heatmap by Vertical and Reasons ---
        st.subheader("🔥 Heatmap of Top Reasons by Vertical")
        heatmap_data = df_filtered.pivot_table(
            index='product_detail_cms_vertical',
            columns='detailed_pv_sub_reasons',
            aggfunc='size',
            fill_value=0
        )

        plt.figure(figsize=(20, 10))
        sns.set_theme(style="whitegrid")
        ax = sns.heatmap(
            heatmap_data,
            cmap="Blues",
            annot=True,
            fmt="d",
            linewidths=0.3,
            cbar_kws={'label': 'Frequency'},
            annot_kws={"size": 8}
        )

        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.title("Top Return Reasons by Vertical", fontsize=18)
        plt.xlabel("Detailed PV Sub-Reasons", fontsize=12)
        plt.ylabel("Vertical", fontsize=12)
        plt.tight_layout()

        st.pyplot(plt)
else:
    st.info("👆 Please upload both dataset and training file to continue.")

