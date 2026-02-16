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

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ✅ MUST be here
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    # ✅ MUST be here
    features = st.multiselect(
        "Select 2 Features for Clustering",
        numeric_columns,
        max_selections=2
    )

    if len(features) == 2:
    X = df[features]
    X = X.fillna(X.mean())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(X_scaled)

    # ✅ Create Cluster column FIRST
    df = df.loc[X.index]
    df["Cluster"] = labels

    # ✅ Now it is SAFE to use Cluster
    st.subheader("Clustered Data")
    st.dataframe(df.head())

    cluster_means = df.groupby("Cluster")[features].mean()

else:
    st.warning("Please select exactly 2 features")
    
        # 🔥 Development Mapping MUST BE HERE
    cluster_means = df.groupby("Cluster")[features].mean()

    sorted_clusters = cluster_means.sort_values(by=features[0]).index.tolist()

    category_map = {}

    if len(sorted_clusters) >= 3:
        category_map[sorted_clusters[0]] = "Underdeveloped"
        category_map[sorted_clusters[1]] = "Developing"
        category_map[sorted_clusters[2]] = "Developed"

    category_map[-1] = "Outlier"

    df["Development_Status"] = df["Cluster"].map(category_map)

    st.subheader("Country Development Classification")
    st.dataframe(df[[features[0], features[1], "Development_Status"]].head(10))











