import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px

st.set_page_config(page_title="World Development Clustering", layout="wide")
st.title("🌍 World Development Clustering App")

# File Upload
uploaded_file = st.file_uploader("Upload 'world_development.csv'", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # --- DATA CLEANING ---
    def clean_numeric(value):
        if isinstance(value, str):
            clean_val = value.replace('$', '').replace(',', '').replace('%', '').strip()
            try:
                return float(clean_val)
            except ValueError:
                return np.nan
        return value

    # Fix the object columns found in your dataset
    cols_to_fix = ['GDP', 'Health Exp/Capita', 'Business Tax Rate', 'Tourism Inbound', 'Tourism Outbound']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = df[col].apply(clean_numeric)

    # Select only numeric columns and drop rows with missing values for the model
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    
    if not numeric_df.empty:
        st.sidebar.header("DBSCAN Parameters")
        eps = st.sidebar.slider("Epsilon (Neighborhood Distance)", 0.1, 10.0, 3.0)
        min_samp = st.sidebar.slider("Min Samples (Cluster Density)", 2, 20, 5)

        # --- MACHINE LEARNING PIPELINE ---
        # 1. Scaling (Essential so GDP doesn't outweigh Birth Rate)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)

        # 2. DBSCAN Clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samp)
        clusters = dbscan.fit_predict(scaled_data)
        numeric_df['Cluster'] = clusters.astype(str) # Convert to string for better plotting colors

        # 3. PCA for 2D Visualization
        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(scaled_data)
        numeric_df['PCA1'] = pca_components[:, 0]
        numeric_df['PCA2'] = pca_components[:, 1]

        # --- VISUALIZATION ---
        st.subheader("Interactive Development Clusters")
        # Link back to original Country names if they exist
        if 'Country' in df.columns:
            numeric_df['Country'] = df.loc[numeric_df.index, 'Country']

        fig = px.scatter(
            numeric_df, x='PCA1', y='PCA2', color='Cluster',
            hover_data=['Country', 'GDP', 'Life Expectancy Female'],
            title="PCA Projection of World Development Metrics",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        st.plotly_chart(fig, use_container_width=True)

        st.write(f"✅ Found {len(numeric_df['Cluster'].unique()) - (1 if '-1' in numeric_df['Cluster'].values else 0)} clusters.")
        st.info("Cluster '-1' represents outlier countries that don't fit into a dense group.")
    else:
        st.error("The dataset contains too many missing values. Try a cleaner version of the file.")
