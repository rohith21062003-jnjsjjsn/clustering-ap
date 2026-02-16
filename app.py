import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

st.title("DBSCAN Clustering Application")

# Upload file
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Original Dataset")
    st.dataframe(df.head())

    # Select features
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    features = st.multiselect("Select 2 Features for Clustering", numeric_cols)

    if len(features) == 2:
        X = df[features]

        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        clusters = dbscan.fit_predict(X_scaled)

        # Add cluster column
        df["Cluster"] = clusters

        st.subheader("Clustered Data")
        st.dataframe(df[[features[0], features[1], "Cluster"]].head(10))

        # Visualization
        st.subheader("Cluster Visualization")
        plt.figure()
        plt.scatter(
            X[features[0]],
            X[features[1]],
            c=clusters,
            cmap="tab10"
        )
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        st.pyplot(plt)
