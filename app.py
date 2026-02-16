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

    # Read File
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Original Dataset")
    st.dataframe(df.head())

    # Select Features
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    features = st.multiselect(
        "Select 2 Features for Clustering",
        numeric_columns,
        max_selections=2
    )

    if len(features) == 2:

        X = df[features]

        # Handle Missing Values
        X = X.fillna(X.mean())

        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # DBSCAN Model
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        clusters = dbscan.fit_predict(X_scaled)

        # Add Cluster Column
        df = df.loc[X.index]
        df["Cluster"] = clusters

        # Show Cluster Preview
        st.subheader("Clustered Data Preview")
        st.dataframe(df[[features[0], features[1], "Cluster"]].head(10))

        # Show Cluster Summary
        st.subheader("Cluster Summary")
        st.write(df["Cluster"].value_counts())

        # Plot
        st.subheader("Cluster Visualization")

        fig, ax = plt.subplots()

        scatter = ax.scatter(
            df[features[0]],
            df[features[1]],
            c=df["Cluster"]
        )

        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.set_title("DBSCAN Clustering")

        st.pyplot(fig)

    else:
        st.warning("Please select exactly 2 features.")
