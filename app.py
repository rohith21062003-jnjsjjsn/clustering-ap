import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="DBSCAN Clustering App", layout="wide")

st.title("DBSCAN Clustering Application")

# File Upload
uploaded_file = st.file_uploader(
    "Upload CSV or Excel File",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)

    # Get numeric columns
numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

# Feature selection
features = st.multiselect(
    "Select exactly 2 features for clustering",
    numeric_columns,
    max_selections=2
)

# Clustering
if len(features) == 2:
    X = df[features]

    # Handle missing values
    X = X.fillna(X.mean())

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(X_scaled)

    # Add cluster column
    df = df.loc[X.index]
    df["Cluster"] = labels

    st.success("Clustering completed successfully")
    st.dataframe(df.head())

else:
    st.warning("Please select exactly 2 numeric features")
    
        # 🔥 Development Mapping MUST BE HERE
    cluster_means = df.groupby("Cluster")[features].mea

    category_map = {}

    if len(sorted_clusters) >= 3:
        category_map[sorted_clusters[0]] = "Underdeveloped"
        category_map[sorted_clusters[1]] = "Developing"
        category_map[sorted_clusters[2]] = "Developed"

    category_map[-1] = "Outlier"

    df["Development_Status"] = df["Cluster"].map(category_map)

    st.subheader("Country Development Classification")
    st.dataframe(df[[features[0], features[1], "Development_Status"]].head(10))
















