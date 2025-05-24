import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Return Reason Splitter & Visualizer")

@st.cache_data
def load_training_data():
    df_train = pd.read_csv("TrainingData.csv")
    df_train['detailed_pv_sub_reasons'] = df_train['detailed_pv_sub_reasons'].str.strip().str.upper()
    df_train['Conclusion'] = df_train['Conclusion'].str.strip().str.upper()
    return dict(zip(df_train['detailed_pv_sub_reasons'], df_train['Conclusion']))

uploaded_file = st.file_uploader("Upload dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    exclusion_destinations = {
        "returns_center_bulk", "bin", "dpv_pending_bulk", "product_missing_bulk"
    }
    inventorized_destinations = {
        "store", "returns_supplier_return_area", "seller_return_area"
    }
    refinishing_destinations = {
        "refinishing", "refurbishment_area"
    }

    required_cols = ['rvp_rto_status', 'business_unit', 'destination_area', 'alpha_flag']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Dataset missing one or more required columns: {required_cols}")
    else:
        # RTO and RVP filtering for RI calculation (same as before)
        valid_rto = df[
            (df['rvp_rto_status'].str.lower() == 'rto') &
            (df['business_unit'].str.lower() != 'giftcard') &
            (~df['destination_area'].isin(exclusion_destinations))
        ].copy()

        valid_rvp = df[
            ((df['alpha_flag'] == 1) | (df['alpha_flag'] == '1')) &
            (df['business_unit'].str.lower() == 'lifestyle') &
            (~df['destination_area'].isin(exclusion_destinations))
        ].copy()

        def inventorized_sum(dest):
            if dest in inventorized_destinations:
                return 1
            if dest in refinishing_destinations:
                return 0.9
            return 0

        rto_inventorized = valid_rto['destination_area'].apply(inventorized_sum).sum()
        rto_total = len(valid_rto)
        rto_ri = (rto_inventorized / rto_total) if rto_total > 0 else 0

        rvp_inventorized = valid_rvp['destination_area'].apply(inventorized_sum).sum()
        rvp_total = len(valid_rvp)
        rvp_ri = (rvp_inventorized / rvp_total) if rvp_total > 0 else 0

        # Show RI boxes
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div style="border:2px solid #4CAF50; border-radius:10px; height:150px; display:flex; flex-direction:column; justify-content:center; align-items:center; background:#E8F5E9;">
                <h3 style="color:#2E7D32; margin:0;">RTO RI</h3>
                <p style="font-size:36px; font-weight:bold; margin:0; color:black;">{rto_ri*100:.2f}%</p>
                <small style="color:black;">Total: {rto_total}, Inventorized: {rto_inventorized:.1f}</small>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style="border:2px solid #4CAF50; border-radius:10px; height:150px; display:flex; flex-direction:column; justify-content:center; align-items:center; background:#E8F5E9;">
                <h3 style="color:#2E7D32; margin:0;">RVP RI</h3>
                <p style="font-size:36px; font-weight:bold; margin:0; color:black;">{rvp_ri*100:.2f}%</p>
                <small style="color:black;">Total: {rvp_total}, Inventorized: {rvp_inventorized:.1f}</small>
            </div>
            """, unsafe_allow_html=True)

        # Expand detailed reasons for full dataset
        df['detailed_pv_sub_reasons'] = df['detailed_pv_sub_reasons'].fillna("").astype(str)
        df_expanded = df.drop('detailed_pv_sub_reasons', axis=1).join(
            df['detailed_pv_sub_reasons'].str.split(',', expand=True)
              .stack().reset_index(level=1, drop=True)
              .rename('detailed_pv_sub_reasons')
        )
        df_expanded['detailed_pv_sub_reasons'] = df_expanded['detailed_pv_sub_reasons'].str.strip().str.upper()

        # Map reasons to conclusions for full dataset
        mapping = load_training_data()
        df_expanded['Conclusion'] = df_expanded['detailed_pv_sub_reasons'].map(mapping)
        df_expanded['Conclusion'] = df_expanded.apply(
            lambda row: row['detailed_pv_sub_reasons'] if pd.isna(row['Conclusion']) else row['Conclusion'], axis=1
        )

        # Normalize Conclusion to avoid duplicates like "MAJORDENTSCRATCH ON PACKAGING"
        df_expanded['Conclusion'] = (
            df_expanded['Conclusion']
            .str.strip()
            .str.upper()
            .str.replace(r'\s+', ' ', regex=True)
        )

        # For heatmap, use valid_rto and valid_rvp but only exclude exclusion_destinations
        # (do NOT exclude inventorized/refinishing destinations)
        def filter_for_heatmap(data):
            return data[~data['destination_area'].isin(exclusion_destinations)].copy()

        rto_heatmap_data = filter_for_heatmap(valid_rto)
        rvp_heatmap_data = filter_for_heatmap(valid_rvp)

        def prepare_for_heatmap(data):
            data['detailed_pv_sub_reasons'] = data['detailed_pv_sub_reasons'].fillna("").astype(str)
            data_exp = data.drop('detailed_pv_sub_reasons', axis=1).join(
                data['detailed_pv_sub_reasons'].str.split(',', expand=True)
                    .stack().reset_index(level=1, drop=True)
                    .rename('detailed_pv_sub_reasons')
            )
            data_exp['detailed_pv_sub_reasons'] = data_exp['detailed_pv_sub_reasons'].str.strip().str.upper()

            data_exp['Conclusion'] = data_exp['detailed_pv_sub_reasons'].map(mapping)
            data_exp['Conclusion'] = data_exp.apply(
                lambda row: row['detailed_pv_sub_reasons'] if pd.isna(row['Conclusion']) else row['Conclusion'], axis=1
            )

            # Normalize Conclusion here as well
            data_exp['Conclusion'] = (
                data_exp['Conclusion']
                .str.strip()
                .str.upper()
                .str.replace(r'\s+', ' ', regex=True)
            )

            # Remove no issue reasons
            no_issues = {'no_issue', 'no issue', 'noissue', 'no_issues', 'no issues', 'no-issue', 'nan', ''}
            data_exp = data_exp[~data_exp['detailed_pv_sub_reasons'].str.lower().isin(no_issues)]

            return data_exp

        def plot_heatmap(df_data, title):
            if df_data.empty or 'Conclusion' not in df_data.columns:
                st.warning(f"No data available for {title}")
                return

            top_conclusions = df_data['Conclusion'].value_counts().nlargest(10).index
            top_verticals = df_data['product_detail_cms_vertical'].value_counts().nlargest(20).index

            df_filtered = df_data[
                df_data['Conclusion'].isin(top_conclusions) &
                df_data['product_detail_cms_vertical'].isin(top_verticals)
            ]

            heatmap_data = pd.crosstab(df_filtered['product_detail_cms_vertical'], df_filtered['Conclusion'])

            if heatmap_data.empty:
                st.warning(f"No data for heatmap: {title}")
                return

            fig, ax = plt.subplots(figsize=(20, 10))
            sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu", linewidths=0.5, linecolor='gray', ax=ax)
            plt.title(title, fontsize=18)
            plt.xlabel("Conclusion")
            plt.ylabel("Vertical")
            st.pyplot(fig)

        # Buttons to generate heatmaps
        if st.button("Generate RTO Heatmap"):
            plot_heatmap(prepare_for_heatmap(rto_heatmap_data), "RTO - Non-Inventorized - Top 10 Reasons by Top 20 Verticals")

        if st.button("Generate RVP Heatmap"):
            plot_heatmap(prepare_for_heatmap(rvp_heatmap_data), "RVP - Non-Inventorized - Top 10 Reasons by Top 20 Verticals")

        # Show transformed full dataset expanded
        if st.checkbox("Show Transformed Data"):
            st.dataframe(df_expanded)

        # Download transformed data as CSV
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(df_expanded)
        st.download_button(
            label="Download Transformed Data as CSV",
            data=csv,
            file_name='transformed_dataset.csv',
            mime='text/csv',
        )
