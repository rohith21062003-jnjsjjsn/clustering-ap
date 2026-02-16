import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="DBSCAN Clustering", layout="centered")

st.title("DBSCAN Clustering Application")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    features = st.multiselect(
        "Select exactly 2 numeric features",
        numeric_columns,
        max_selections=2
    )

    if len(features) == 2:
        X = df[features].fillna(df[features].mean())

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = DBSCAN(eps=0.5, min_samples=5)
        df["Cluster"] = model.fit_predict(X_scaled)

        st.success("Clustering completed successfully")
        if "Cluster" in df.columns:
            st.subheader("cluster summary")
            cluster_summary=cluster_means = df.groupby("Cluster").mean(numeric_only=True)
            st.dataframe(cluster_summary)

    sorted_clusters = cluster_means.mean(axis=1).sort_values().index

    label_map = {
        sorted_clusters[0]: "Underdeveloped",
        sorted_clusters[1]: "Developing",
        sorted_clusters[-1]: "Developed"
    }

    df["Development_Status"] = df["Cluster"].map(label_map)

    st.dataframe(df[["Cluster", "Development_Status"]].head())
    
        st.dataframe(df.head())

        fig, ax = plt.subplots()
        ax.scatter(df[features[0]], df[features[1]], c=df["Cluster"])
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        st.pyplot(fig)

    else:
        st.warning("Please select exactly 2 numeric features")


















