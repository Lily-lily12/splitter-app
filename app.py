import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="CSV Splitter", layout="centered")
st.title("📂 Split Multi-Value Rows Tool")

uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    with st.spinner("Reading file..."):
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

    col_to_split = "detailed_pv_sub_reasons"

    if col_to_split in df.columns:
        with st.spinner("Processing data..."):
            df[col_to_split] = df[col_to_split].astype(str).str.split(',')
            df_expanded = df.explode(col_to_split)
            df_expanded[col_to_split] = df_expanded[col_to_split].str.strip()

        st.success(f"✅ Split completed! {len(df_expanded)} rows generated.")
        st.write("🔍 Preview (First 10 rows):", df_expanded.head(10))

        output = io.BytesIO()
        df_expanded.to_csv(output, index=False)
        st.download_button(
            label="📥 Download Transformed CSV",
            data=output.getvalue(),
            file_name="transformed_dataset.csv",
            mime="text/csv"
        )
    else:
        st.error(f"❌ Column '{col_to_split}' not found in the uploaded file.")

