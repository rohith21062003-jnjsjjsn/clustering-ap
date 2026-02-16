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

    # Load file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Select numeric columns
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    features = st.multiselect(
        "Select exactly 2 numeric features",
        numeric_columns,
        max_selections=2
    )

    if len(features) == 2:

        # Handle missing values
        X = df[features].fillna(df[features].mean())

        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # DBSCAN
        model = DBSCAN(eps=0.5, min_samples=5)
        df["Cluster"] = model.fit_predict(X_scaled)

        st.success("Clustering completed successfully")
        st.dataframe(df.head())

        # ---- Cluster labeling ----
        if "Cluster" in df.columns:

            cluster_means = df.groupby("Cluster").mean(numeric_only=True)

            df["Development_status"]=
            df["Development_status"].fillna('Noise")

            if len(cluster_means) >= 3:
                sorted_clusters = cluster_means.mean(axis=1).sort_values().index

                label_map = {
                    sorted_clusters[0]: "Underdeveloped",
                    sorted_clusters[1]: "Developing",
                    sorted_clusters[-1]: "Developed"
                }

                df["Development_Status"] = df["Cluster"].map(label_map)
                df["Development_Status"].fillna("Noise", inplace=True)

                st.subheader("Cluster Labels")
                st.dataframe(df[["Cluster", "Development_Status"]].head())

        # ---- Visualization ----
        fig, ax = plt.subplots()
        scatter = ax.scatter(
            df[features[0]],
            df[features[1]],
            c=df["Cluster"],
            cmap="viridis"
        )
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.set_title("DBSCAN Clustering Result")
        st.pyplot(fig)

    else:
        st.warning("Please select exactly 2 numeric features.")





























