import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="CSV Splitter + Heatmap", layout="wide")
st.title("📂 Split Multi-Value Rows Tool + Heatmap Visualizer")

uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    with st.spinner("Reading file..."):
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

    col_to_split = "detailed_pv_sub_reasons"
    pivot_index_col = "product_detail_cms_vertical"

    if col_to_split in df.columns and pivot_index_col in df.columns:
        with st.spinner("Processing data..."):
            df[col_to_split] = df[col_to_split].astype(str).str.split(',')
            df_expanded = df.explode(col_to_split)
            df_expanded[col_to_split] = df_expanded[col_to_split].str.strip()

        st.success(f"✅ Split completed! {len(df_expanded)} rows generated.")
        st.write("🔍 Preview (First 10 rows):", df_expanded.head(10))

        # 🎯 Create pivot table
        pivot_table = pd.pivot_table(
            df_expanded,
            index=pivot_index_col,
            columns=col_to_split,
            aggfunc='size',
            fill_value=0
        )

        st.subheader("📊 Count Table: Defects per Product Vertical")
        st.dataframe(pivot_table, use_container_width=True)

        # 🔥 Plot heatmap
        st.subheader("🔥 Heatmap of Defects by Product Vertical")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(pivot_table, cmap="YlOrRd", annot=True, fmt='d', linewidths=.5, ax=ax)
        st.pyplot(fig)

        # 💾 Download transformed CSV
        output = io.BytesIO()
        df_expanded.to_csv(output, index=False)
        st.download_button(
            label="📥 Download Transformed CSV",
            data=output.getvalue(),
            file_name="transformed_dataset.csv",
            mime="text/csv"
        )
    else:
        st.error("❌ Required columns not found in the uploaded file.")
