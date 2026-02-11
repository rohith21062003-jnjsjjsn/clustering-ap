import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px

st.set_page_config(page_title="World Development Clustering", layout="wide")
st.title("🌍 World Development Clustering App")

# 1. FIX: Flexible File Uploader (Accepts CSV and Excel)
uploaded_file = st.file_uploader("Upload your data file", type=["csv", "xlsx"])

if uploaded_file:
    # Handle the file type correctly
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # --- DATA CLEANING FOR SYMBOLS ($ and ,) ---
    def clean_currency_and_percent(value):
        if isinstance(value, str):
            # Remove symbols that prevent math operations
            clean_val = value.replace('$', '').replace(',', '').replace('%', '').strip()
            try:
                return float(clean_val)
            except ValueError:
                return np.nan
        return value

    # Automatically clean all columns that look like text but should be numbers
    for col in df.columns:
        if df[col].dtype == 'object' and col != 'Country':
            df[col] = df[col].apply(clean_currency_and_percent)

    # Prepare numeric data for DBSCAN (dropping rows with missing values)
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    
    if not numeric_df.empty:
        st.sidebar.header("DBSCAN Parameters")
        eps = st.sidebar.slider("Epsilon (Neighborhood Distance)", 0.1, 10.0, 3.0)
        min_samp = st.sidebar.slider("Min Samples (Cluster Density)", 2, 20, 5)

        # 2. Scaling (Essential for mixed-scale data)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)

        # 3. DBSCAN and PCA Visualization
        dbscan = DBSCAN(eps=eps, min_samples=min_samp)
        clusters = dbscan.fit_predict(scaled_data)
        numeric_df['Cluster'] = clusters.astype(str)

        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(scaled_data)
        numeric_df['PCA1'] = pca_components[:, 0]
        numeric_df['PCA2'] = pca_components[:, 1]

        # Final Plot
        st.subheader("Interactive Clusters")
        if 'Country' in df.columns:
            numeric_df['Country'] = df.loc[numeric_df.index, 'Country']

        fig = px.scatter(
            numeric_df, x='PCA1', y='PCA2', color='Cluster',
            hover_data=['Country'] if 'Country' in numeric_df.columns else None,
            title="Development Clusters (Cleaned Data)"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No numeric data found. Check if your file contains numbers!")
