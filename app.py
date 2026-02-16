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
        st.dataframe(df.head())

        fig, ax = plt.subplots()
        ax.scatter(df[features[0]], df[features[1]], c=df["Cluster"])
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        st.pyplot(fig)

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

















