import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set Streamlit page config
st.set_page_config(page_title="Top Return Reasons Heatmap", layout="wide")

st.title("📊 Heatmap: Top Return Reasons by Business Unit")

# File uploader
uploaded_file = st.file_uploader("Upload your cleaned CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Drop rows with NaN or 'no_issue' in detailed_pv_sub_reasons
    df = df.dropna(subset=['detailed_pv_sub_reasons', 'business_unit'])
    df = df[df['detailed_pv_sub_reasons'].str.lower() != 'no_issue']

    # Filter to Top N most common return reasons
    top_n = 15
    top_reasons = df['detailed_pv_sub_reasons'].value_counts().nlargest(top_n).index
    df_filtered = df[df['detailed_pv_sub_reasons'].isin(top_reasons)]

    # Create pivot table
    heatmap_data = df_filtered.pivot_table(
        index='business_unit',
        columns='detailed_pv_sub_reasons',
        aggfunc='size',
        fill_value=0
    )

    # Plot heatmap
    plt.figure(figsize=(18, 8))
    sns.set_theme(style="whitegrid")

    ax = sns.heatmap(
        heatmap_data,
        cmap="Blues",
        annot=True,
        fmt="d",
        linewidths=0.5,
        cbar_kws={"label": "Frequency"},
        annot_kws={"size": 9}
    )

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title("Top Return Reasons by Business Unit", fontsize=18, pad=20)
    plt.xlabel("Detailed PV Sub-Reasons", fontsize=12)
    plt.ylabel("Business Unit", fontsize=12)
    plt.tight_layout()

    st.pyplot(plt)


