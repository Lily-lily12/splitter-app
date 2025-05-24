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
        "store", "returns_supplier_return_area", "seller_return_area",
        "refinishing", "refurbishment_area"
    }

    required_cols = ['rvp_rto_status', 'business_unit', 'destination_area', 'alpha_flag']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Dataset missing one or more required columns: {required_cols}")
    else:
        # RI calculations
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
            if dest in {"store", "returns_supplier_return_area", "seller_return_area"}:
                return 1
            if dest in {"refinishing", "refurbishment_area"}:
                return 0.9
            return 0

        rto_inventorized = valid_rto['destination_area'].apply(inventorized_sum).sum()
        rto_total = len(valid_rto)
        rto_ri = (rto_inventorized / rto_total) if rto_total > 0 else 0

        rvp_inventorized = valid_rvp['destination_area'].apply(inventorized_sum).sum()
        rvp_total = len(valid_rvp)
        rvp_ri = (rvp_inventorized / rvp_total) if rvp_total > 0 else 0

        # Display RI metrics
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

        # Expand detailed reasons
        df['detailed_pv_sub_reasons'] = df['detailed_pv_sub_reasons'].fillna("").astype(str)
        df_expanded = df.drop('detailed_pv_sub_reasons', axis=1).join(
            df['detailed_pv_sub_reasons'].str.split(',', expand=True).stack().reset_index(level=1, drop=True).rename('detailed_pv_sub_reasons')
        )
        df_expanded['detailed_pv_sub_reasons'] = df_expanded['detailed_pv_sub_reasons'].str.strip().str.upper()

        # Map to conclusions
        mapping = load_training_data()
        df_expanded['Conclusion'] = df_expanded['detailed_pv_sub_reasons'].map(mapping)
        df_expanded['Conclusion'] = df_expanded.apply(
            lambda row: row['detailed_pv_sub_reasons'] if pd.isna(row['Conclusion']) else row['Conclusion'], axis=1
        )

        # Filter unwanted data
        df_expanded = df_expanded[
            (~df_expanded['destination_area'].isin(exclusion_destinations)) &
            (~df_expanded['detailed_pv_sub_reasons'].str.lower().isin([
                'no_issue', 'no issue', 'noissue', 'no_issues', 'no issues', 'no-issue', 'nan', ''
            ])) &
            (~df_expanded['destination_area'].isin(inventorized_destinations))
        ]

        # Set top categories
        top_conclusions = df_expanded['Conclusion'].value_counts().nlargest(10).index
        top_verticals = df_expanded['product_detail_cms_vertical'].value_counts().nlargest(20).index

        def plot_heatmap(filtered_df, title):
            df_viz = filtered_df[
                filtered_df['Conclusion'].isin(top_conclusions) &
                filtered_df['product_detail_cms_vertical'].isin(top_verticals)
            ]
            heatmap_data = pd.crosstab(df_viz['product_detail_cms_vertical'], df_viz['Conclusion'])
            fig, ax = plt.subplots(figsize=(20, 10))
            sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu", linewidths=0.5, linecolor='gray', ax=ax)
            plt.title(title, fontsize=18)
            plt.xlabel("Conclusion")
            plt.ylabel("Vertical")
            st.pyplot(fig)

        if st.checkbox("Generate RTO Heatmap"):
            rto_filtered = df_expanded[
                (df_expanded['rvp_rto_status'].str.lower() == 'rto') &
                (df_expanded['business_unit'].str.lower() != 'giftcard')
            ]
            plot_heatmap(rto_filtered, "RTO - Non-Inventorized - Top 10 Return Reasons by Top 20 Verticals")

        if st.checkbox("Generate RVP Heatmap"):
            rvp_filtered = df_expanded[
                ((df_expanded['alpha_flag'] == 1) | (df_expanded['alpha_flag'] == '1')) &
                (df_expanded['business_unit'].str.lower() == 'lifestyle')
            ]
            plot_heatmap(rvp_filtered, "RVP - Non-Inventorized - Top 10 Return Reasons by Top 20 Verticals")

        if st.checkbox("Show Transformed Data"):
            st.dataframe(df_expanded)

        csv = df_expanded.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Transformed Data as CSV",
            data=csv,
            file_name='transformed_dataset.csv',
            mime='text/csv',
        )


