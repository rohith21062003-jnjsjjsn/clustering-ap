import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="DBSCAN Clustering App", layout="wide")

st.title("📊 DBSCAN Clustering Application")

# -----------------------------
# Upload file
# -----------------------------
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.subheader("🔍 Original Dataset")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.subheader("⚙️ Feature Selection")
    selected_cols = st.multiselect(
        "Select numeric columns for clustering",
        numeric_cols,
        default=numeric_cols[:2]
    )

    if len(selected_cols) >= 2:

        eps = st.slider("Epsilon (eps)", 0.1, 5.0, 0.5)
        min_samples = st.slider("Min Samples", 1, 10, 5)

        df_numeric = df[selected_cols].dropna()

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_numeric)

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(scaled_data)

        df_numeric["Cluster"] = labels

        st.subheader("📌 Cluster Counts")
        st.write(df_numeric["Cluster"].value_counts())

        noise_points = (df_numeric["Cluster"] == -1).sum()
        st.success(f"Number of Noise Points (-1): {noise_points}")

        # -----------------------------
        # Cluster Visualization
        # -----------------------------
        st.subheader("📈 Cluster Visualization")

        x_axis = st.selectbox("X-axis", selected_cols, index=0)
        y_axis = st.selectbox("Y-axis", selected_cols, index=1)

        fig, ax = plt.subplots()
        scatter = ax.scatter(
            df_numeric[x_axis],
            df_numeric[y_axis],
            c=df_numeric["Cluster"]
        )

        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_title("DBSCAN Cluster Scatter Plot")

        st.pyplot(fig)

        st.subheader("📄 Clustered Data")
        st.dataframe(df_numeric)

    else:
        st.warning("Please select at least two numeric columns.")
