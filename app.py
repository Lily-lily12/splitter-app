import pandas as pd
import streamlit as st

# ✅ Correct raw GitHub CSV URL
GITHUB_RAW_CSV_URL = "https://raw.githubusercontent.com/Lily-lily12/splitter-app/main/TrainingData_AUGMENTED.csv"

# Title
st.title("🧠 Casper Conclusion Heatmap + RI Percentages")

# Load training data from GitHub
@st.cache_data
def load_training_data():
    df = pd.read_csv(GITHUB_RAW_CSV_URL)
    return df

df = load_training_data()

# Upload mapping CSV
mapping_file = st.file_uploader("📁 Upload Mapping File (CSV)", type=["csv"])

if mapping_file:
    # Load and clean mapping
    mapping_df = pd.read_csv(mapping_file)
    mapping_df.columns = mapping_df.columns.str.strip().str.lower()
    mapping_df['detailed reason'] = mapping_df['detailed reason'].str.strip().str.upper()
    mapping_df['conclusion'] = mapping_df['conclusion'].str.strip().str.upper()

    # Create mapping dictionary
    reason_to_conclusion = dict(zip(mapping_df['detailed reason'], mapping_df['conclusion']))

    # Preprocess and explode the reasons
    df['detailed_pv_sub_reasons'] = df['detailed_pv_sub_reasons'].fillna('').str.upper().str.split(',')
    exploded_df = df.explode('detailed_pv_sub_reasons')
    exploded_df['detailed_pv_sub_reasons'] = exploded_df['detailed_pv_sub_reasons'].str.strip()

    # Map to conclusion
    exploded_df['conclusion'] = exploded_df['detailed_pv_sub_reasons'].map(reason_to_conclusion)
    exploded_df = exploded_df[exploded_df['conclusion'].notna()]
    exploded_df['count'] = 1

    # Pivot table for heatmap
    heatmap_df = exploded_df.pivot_table(
        index='casper_id',
        columns='conclusion',
        values='count',
        aggfunc='sum',
        fill_value=0
    )
    heatmap_df = heatmap_df.reindex(sorted(heatmap_df.columns), axis=1)

    # Show pivot table
    st.write("✅ Cleaned Pivot Table:")
    st.dataframe(heatmap_df)

    # Display Heatmap
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt

        st.write("📊 Heatmap:")
        fig, ax = plt.subplots(figsize=(15, len(heatmap_df) * 0.4 + 1))
        sns.heatmap(heatmap_df, annot=True, fmt="d", cmap="YlGnBu", cbar=True, ax=ax)
        st.pyplot(fig)
    except ImportError:
        st.warning("Install seaborn and matplotlib to enable heatmap rendering.")

    # ➕ RI Percentages
    st.markdown("### 📈 RI Percentages")

    # RTO RI %
    rto_df = df[df['project_name'].str.upper().str.contains("RTO", na=False)]
    rto_ri = rto_df['is_ri'].sum()
    rto_total = rto_df['is_ri'].count()
    rto_pct = (rto_ri / rto_total * 100) if rto_total > 0 else 0
    st.write(f"🔹 **RTO RI %:** {rto_pct:.2f}%  ({rto_ri}/{rto_total})")

    # RVP RI %
    rvp_df = df[df['project_name'].str.upper().str.contains("RVP", na=False)]
    rvp_ri = rvp_df['is_ri'].sum()
    rvp_total = rvp_df['is_ri'].count()
    rvp_pct = (rvp_ri / rvp_total * 100) if rvp_total > 0 else 0
    st.write(f"🔸 **RVP RI %:** {rvp_pct:.2f}%  ({rvp_ri}/{rvp_total})")

